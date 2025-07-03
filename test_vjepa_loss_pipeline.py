#!/usr/bin/env python3

"""
Test script to validate calculate_torch_vjepa_loss function by reproducing
the IntPhys evaluation results from reproduce_intphys_clean.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import json
from einops import rearrange
import importlib
from torch.nn.functional import pad
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Add Quentin's codebase to path
sys.path.insert(0, '/home/yjianhao/project/quentinecode/vjepa2')

# Import Quentin's modules
from app.vjepa.transforms import make_transforms

# Import our loss function
from compute_vjepa_score import calculate_torch_vjepa_loss

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_breaking_points(clip):
    bps = []
    for diff in [clip[0]-clip[1],clip[0]-clip[2],clip[0]-clip[3]]:
        try:
            i = torch.argwhere(diff.sum(2).sum(2).sum(0)!=0)[0,0].item()
        except:
            i = clip.shape[2]
        bps.append(i)
    return bps

def get_matches(bps):
    if np.argmax(bps) == 0:
        return [[0,1],[2,3]]
    elif np.argmax(bps) == 1:
        return [[0,2],[1,3]]
    else:
        return [[0,3],[1,2]]

def compute_metrics(losses, labels):
    """Exact reproduction of Quentin's compute_metrics function"""
    
    metrics = {}
    logger.info("Computing metrics")
    
    # Split by labels (1=possible/real, 0=impossible/fake)  
    loss_real = losses[torch.where(labels == 1)]
    loss_fake = losses[torch.where(labels == 0)]
    
    logger.info(f"Real losses shape: {loss_real.shape}")
    logger.info(f"Fake losses shape: {loss_fake.shape}")

    # Relative accuracy metrics - key IntPhys metrics
    acc_pairwise_mean = (loss_real.mean(1) < loss_fake.mean(1)).sum() / loss_real.shape[0] * 100
    acc_pairwise_max = (loss_real.max(1)[0] < loss_fake.max(1)[0]).sum() / loss_real.shape[0] * 100

    metrics["Relative Accuracy (avg)"] = acc_pairwise_mean.item()
    metrics["Relative Accuracy (max)"] = acc_pairwise_max.item()

    # Absolute accuracy metrics
    data1 = loss_real.max(1)[0]  # possible videos max loss
    data2 = loss_fake.max(1)[0]  # impossible videos max loss
    
    thresh = 0  # Original uses 0 threshold
    accuracy_abs = ((data1 < thresh).sum() + (data2 > thresh).sum()) / (data1.shape[0] + data2.shape[0]) * 100

    metrics["Absolute Accuracy (max)"] = accuracy_abs.item()
    metrics["Classifier threshold"] = thresh

    # Best absolute accuracy 
    threshs = np.linspace(data1.min().item(), data2.max().item(), 100)
    accs = []
    for thresh in threshs:
        accs.append(((data1 < thresh).sum() + (data2 > thresh).sum()) / (data1.shape[0] + data2.shape[0]))
    best_accuracy_abs = torch.max(torch.Tensor(accs)) * 100
    oracle_thresh = threshs[torch.argmax(torch.Tensor(accs))]

    metrics["Best Absolute Accuracy (max)"] = best_accuracy_abs.item()
    metrics["Best Classifier threshold"] = oracle_thresh

    # AUPRC - Area under precision-recall curve
    precision_max, recall_max, _ = precision_recall_curve(labels.cpu().numpy(), -losses.max(1)[0].cpu().numpy())
    precision_mean, recall_mean, _ = precision_recall_curve(labels.cpu().numpy(), -losses.mean(1).cpu().numpy())
    auprc_max = auc(recall_max, precision_max)
    auprc_mean = auc(recall_mean, precision_mean)

    metrics["AUPRC (avg)"] = auprc_mean
    metrics["AUPRC (max)"] = auprc_max

    # AUROC - Area under ROC curve
    fpr_max, tpr_max, _ = roc_curve(labels.cpu().numpy(), -losses.max(1)[0].cpu().numpy())
    fpr_mean, tpr_mean, _ = roc_curve(labels.cpu().numpy(), -losses.mean(1).cpu().numpy())
    auroc_max = auc(fpr_max, tpr_max)
    auroc_mean = auc(fpr_mean, tpr_mean)

    metrics["AUROC (avg)"] = auroc_mean
    metrics["AUROC (max)"] = auroc_max

    return metrics

class IntPhysDataset(torch.utils.data.Dataset):
    """Exact reproduction of Quentin's IntPhysDataset"""
    
    def __init__(self, data_path, frames_per_clip=16, frame_step=4, transform=None, shared_transform=None):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.transform = transform
        self.shared_transform = shared_transform
        
        self.scenes = sorted(os.listdir(self.data_path))
        self.length_clip = self.frames_per_clip * self.frame_step

    def __getitem__(self, index):
        scene = self.scenes[index]
        # To change if  we want to get a precise one/all of them
        labels = []
        buffer = []
        paths = []
        for possibility in [1,2,3,4]:
            frames_all = sorted(os.listdir(f"{self.data_path}/{scene}/{possibility}/scene"))

            start = np.random.randint(0,len(frames_all)-self.length_clip)
            if self.length_clip > 90 :
                start = 0
            frames_to_load = frames_all[start:start+self.length_clip:self.frame_step]
            
            #get
            clip = []
            for frame in frames_to_load:
                frame_ = Image.open(f"{self.data_path}/{scene}/{possibility}/scene/{frame}")
                frame_ = torch.Tensor(np.array(frame_))
                clip.append(frame_)
            clip = torch.stack(clip)
            if self.transform is not None:
                clip = self.transform(clip)
            buffer.append(clip)
            try:
                with open(f"{self.data_path}/{scene}/{possibility}/status.json",'r') as f:
                    a = json.load(f)
                label  = a['header']['is_possible']
            except:
                label = True
            labels.append(label)

            paths.append(f"{self.data_path[-2:]}/{scene}/{possibility}")
        
        buffer = torch.stack(buffer)
        paths = torch.tensor(buffer)
        labels = torch.Tensor(labels)
        id = torch.Tensor([index])

        return buffer,labels, id
    
    def __len__(self):
        return len(self.scenes)

def init_model(checkpoint_path, device):
    """Initialize model exactly as Quentin does"""
    
    # Model configuration from Quentin's config
    model_kwargs = {
        'resolution': 256,
        'encoder': {
            'model_name': 'vit_huge',
            'checkpoint_key': 'encoder',
            'is_causal': False,
            'local_window': [-1, -1, -1],
            'uniform_power': True,
            'use_activation_checkpointing': True,
            'use_mask_tokens': True,
            'use_rope': True,
            'zero_init_mask_tokens': True,
            'num_frames': 16,
        },
        'target_encoder': {
            'checkpoint_key': 'target_encoder',
        },
        'predictor': {
            'model_name': 'vit_predictor',
            'checkpoint_key': 'predictor',
            'depth': 12,
            'is_causal': False,
            'local_window': [-1, -1, -1],
            'num_heads': 12,
            'uniform_power': True,
            'use_activation_checkpointing': True,
            'use_mask_tokens': True,
            'use_rope': True,
            'zero_init_mask_tokens': True,
            'num_mask_tokens': 10,
            'num_frames': 16,
        }
    }
    
    wrapper_kwargs = {
        'no_predictor': False,
    }
    
    # Import and initialize model exactly as Quentin does
    module_name = 'app.vjepa.modelcustom.vit_encoder_predictor_noar_targets'
    model = importlib.import_module(module_name).init_module(
        frames_per_clip=16,
        nb_context_frames=1,  # Will be updated dynamically
        checkpoint=checkpoint_path,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    return model

def extract_losses_for_scene_using_our_function(model, scene_data, context_lengths, frames_per_clip=16, stride=2, use_bfloat16=True, device='cuda'):
    """Extract losses for a single scene using our calculate_torch_vjepa_loss function"""
    
    labels = scene_data[1]  # Shape should be [4] for 4 possibilities
    clip = scene_data[0].to(device)  # Shape should be [4, C, T, H, W]
    
    num_videos = clip.shape[0]
    logger.info(f"Video shape {clip.shape}")
    
    # Use breaking points for IntPhys exactly as Quentin does
    bps = get_breaking_points(clip)
    matches = get_matches(bps)
    logger.info(f"Breaking points: {bps}")
    logger.info(f"Matches: {matches}")
    
    all_losses_ctxt = []
    for CTXT_LEN in context_lengths:
        logger.info("=" * 40)
        logger.info(f"Context length: {CTXT_LEN}")
        
        # Process each video using our calculate_torch_vjepa_loss function
        video_losses = []
        for video_idx in range(num_videos):
            single_video = clip[video_idx:video_idx+1]  # Keep batch dimension
            logger.info(f"Processing video {video_idx} with shape {single_video.shape}")
            
            # Use our loss function to get individual window losses
            _, window_losses = calculate_torch_vjepa_loss(
                single_video, 
                model, 
                context_length=CTXT_LEN,
                frames_per_clip=frames_per_clip,
                stride=stride,
                use_bfloat16=use_bfloat16,
                require_grad=False,
                mode='max',  # This determines final aggregation, but we get individual losses too
                return_arr=True,  # This gives us individual window losses
                is_vae_output=False  # IntPhys data is already in correct format after transforms
            )
            
            # window_losses is shape [num_videos, num_windows] 
            # Extract for single video: [num_windows]
            video_loss = window_losses.squeeze(0)  # Remove batch dimension
            video_losses.append(video_loss)
            
            logger.info(f"Video {video_idx} loss shape: {video_loss.shape}")
        
        # Pad to same length and stack
        max_length = max([l.size(0) for l in video_losses])
        padded_losses = []
        for l in video_losses:
            if l.size(0) < max_length:
                padding = torch.zeros(max_length - l.size(0)).to(l.device)
                l = torch.cat([l, padding], dim=0)
            padded_losses.append(l)
        
        losses = torch.stack(padded_losses)  # [num_videos, max_length]
        logger.info(f"Context {CTXT_LEN} losses shape: {losses.shape}")
        all_losses_ctxt.append(losses)
    
    # Stack across context lengths
    losses = torch.stack(all_losses_ctxt)  # [num_contexts, num_videos, max_length]
    losses = losses.permute(1, 0, 2)  # [num_videos, num_contexts, max_length]
    
    # Extract matched pairs exactly as Quentin does
    scene_losses = []
    scene_labels = []
    for match in matches:
        pair_losses = losses[match]  # Shape: [2, num_contexts, max_length]
        pair_labels = labels[match]  # Shape: [2] 
        
        # Add both possible arrangements to get pairwise data
        scene_losses.append(pair_losses)  
        scene_labels.append(pair_labels)
    
    return scene_losses, scene_labels

def evaluate_intphys_with_our_loss(data_path, checkpoint_path, device='cuda', context_lengths=[2,4,6,8,10]):
    """Evaluate IntPhys using our calculate_torch_vjepa_loss function"""
    
    logger.info(f"Starting IntPhys evaluation with our loss function")
    
    # Initialize transform exactly as Quentin does
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1/1, 1/1],  # No aspect ratio change for IntPhys
        random_resize_scale=[1.0, 1.0],  # No scale change for IntPhys
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=256
    )
    
    # Initialize model
    model = init_model(checkpoint_path, device)
    
    # Initialize dataset
    dataset = IntPhysDataset(
        data_path=data_path,
        frames_per_clip=99//2,  # frame_step=2, so this gives us the right number
        frame_step=2,
        transform=transform
    )
    
    logger.info(f"Found {len(dataset.scenes)} scenes: {dataset.scenes}")
    
    all_losses = []
    all_labels = []
    all_ids = []
    
    # Process all scenes
    for scene_idx in range(len(dataset)):
        scene_name = dataset.scenes[scene_idx]
        logger.info(f"Processing scene {scene_idx+1}/{len(dataset)}: {scene_name}")
        
        # Get scene data
        scene_data = dataset[scene_idx]
        
        # Extract losses using our function
        scene_losses, scene_labels = extract_losses_for_scene_using_our_function(
            model, scene_data, context_lengths, 
            frames_per_clip=16, stride=2, use_bfloat16=True, device=device
        )
        
        all_losses.extend(scene_losses)
        all_labels.extend(scene_labels) 
        all_ids.append(torch.tensor([scene_idx]))
    
    # Format for compute_metrics exactly as original
    lengths = []
    for pair_losses in all_losses:
        lengths.append(pair_losses.size(-1))
    max_length = max(lengths)
    
    # Pad and concatenate all pair losses
    formatted_losses = []
    formatted_labels = []
    
    for pair_losses, pair_labels in zip(all_losses, all_labels):
        # Pad if needed
        if pair_losses.size(-1) < max_length:
            padding = torch.zeros(pair_losses.size(0), pair_losses.size(1), max_length - pair_losses.size(-1)).to(pair_losses.device)
            pair_losses = torch.cat([pair_losses, padding], dim=2)
        
        # Add both videos in the pair
        formatted_losses.append(pair_losses)  # [2, num_contexts, max_length]
        formatted_labels.append(pair_labels)  # [2]
    
    # Concatenate all pairs
    all_losses = torch.cat(formatted_losses, dim=0)  # [total_videos, num_contexts, max_length]
    all_labels = torch.cat(formatted_labels, dim=0)  # [total_videos]
    
    # Compute results for each context length
    results = {}
    for i, context_len in enumerate(context_lengths):
        losses = all_losses[:, i, :]  # [total_videos, max_length]
        labels = all_labels  # [total_videos]
        
        logger.info(f"Context {context_len}: losses shape {losses.shape}, labels shape {labels.shape}")
        
        # Compute metrics exactly as Quentin does
        metrics = compute_metrics(losses, labels)
        
        results[f'context_{context_len}'] = metrics
        
        logger.info(f"Context {context_len} metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
    
    return results

if __name__ == "__main__":
    # Test evaluation using our loss function
    data_path = "/home/yjianhao/project/video_guidance/dev/O1"  # Point directly to O1 folder
    checkpoint_path = "/home/yjianhao/project/quentinecode/vjepa2/vit-h-open/vith.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Test with a subset first
    results = evaluate_intphys_with_our_loss(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        device=device,
        context_lengths=[2, 4]  # Start with just 2 context lengths for testing
    )
    
    print(f"\n" + "="*80)
    print("RESULTS USING OUR LOSS FUNCTION")
    print("="*80)
    for context, result in results.items():
        print(f"{context}:")
        print(f"  Relative Accuracy (avg): {result['Relative Accuracy (avg)']:.2f}%")
        print(f"  Relative Accuracy (max): {result['Relative Accuracy (max)']:.2f}%")
        print(f"  AUPRC (avg): {result['AUPRC (avg)']:.6f}")
        print(f"  AUPRC (max): {result['AUPRC (max)']:.6f}")
        print(f"  AUROC (avg): {result['AUROC (avg)']:.6f}")
        print(f"  AUROC (max): {result['AUROC (max)']:.6f}")
        print()
    
    # Save results for comparison
    import json
    output_file = "intphys_o1_results_our_loss.json"
    save_results = {}
    for context, result in results.items():
        save_results[context] = result
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

print("Script created successfully!") 