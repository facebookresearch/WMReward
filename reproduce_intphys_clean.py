#!/usr/bin/env python3

"""
Clean reproduction of Quentin's V-JEPA IntPhys evaluation.
Faithfully follows the exact implementation from /home/yjianhao/project/quentinecode/vjepa2
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

def pad_tensors(tensors,max_length,length_axis=-1):
    padded_tensors = []
    for t in tensors:
        padding_needed = max_length - t.size(length_axis)
        padded_tensors.append(pad(t, (0, padding_needed)))
    return padded_tensors

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

def extract_losses_for_scene(model, scene_data, context_lengths, frames_per_clip=16, stride=2, use_bfloat16=True, device='cuda'):
    """Extract losses for a single scene, exactly as Quentin does"""
    
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
        
        # Update model parameters exactly as Quentin does
        model.nb_context_frames = CTXT_LEN
        model.frames_per_clip = frames_per_clip
        
        # Set grid_depth exactly as Quentin does
        model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
        
        logger.info(f"CTXT {CTXT_LEN}, FPC = {model.frames_per_clip}, grid_depth = {model.grid_depth}")
        
        # Create sliding windows exactly as Quentin does  
        pieces = clip.unfold(2, model.frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
        pieces = pieces.flatten(0, 1)
        pieces = rearrange(pieces, "b t c h w -> b c t h w")

        # Save the first piece and check the value range
        first_piece = pieces[0].detach().cpu()
        logger.info(f"First piece shape: {first_piece.shape}")
        logger.info(f"First piece value range: [{first_piece.min():.6f}, {first_piece.max():.6f}]")
        logger.info(f"First piece mean: {first_piece.mean():.6f}")
        logger.info(f"First piece std: {first_piece.std():.6f}")
        # import pdb; pdb.set_trace()        
        logger.info(f"pieces {pieces.shape}")
        
        # Process in chunks exactly as Quentin does
        chunked_preds = []
        chunked_targets = []
        CHUNK_SIZE = 1 # Same as Quentin's
        
        logger.info(f"Number of chunks {int(np.ceil(pieces.shape[0]/CHUNK_SIZE))}")
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
                chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]
                
                preds, targets = model(chunk)
                chunked_preds.append(preds.cpu())
                chunked_targets.append(targets.cpu())
        
        preds = torch.vstack(chunked_preds)
        targets = torch.vstack(chunked_targets)
        preds = preds.view(num_videos, -1, *preds.shape[1:])
        targets = targets.view(num_videos, -1, *targets.shape[1:])
        
        logger.info(f"Targets: {targets.shape}")
        logger.info(f"Preds: {preds.shape}")
        
        # Compute loss exactly as Quentin does
        loss = F.l1_loss(preds, targets, reduction="none").mean((2, 3)).detach().to(device)
        logger.info(f"Loss: {loss.shape}")
        logger.info(f"Loss values: {loss}")
        
        all_losses_ctxt.append(loss)
    
    # Pad tensors to same length exactly as Quentin does
    max_length = max([l.size(-1) for l in all_losses_ctxt])
    padded_losses = []
    for l in all_losses_ctxt:
        if l.size(-1) < max_length:
            padding = torch.zeros(l.size(0), max_length - l.size(-1)).to(l.device)
            l = torch.cat([l, padding], dim=1)
        padded_losses.append(l)
    
    losses = torch.stack(padded_losses)
    losses = losses.permute(1, 0, 2)  # [num_videos, num_contexts, max_length]
    
    # Extract matched pairs exactly as Quentin does - this is the key fix!
    scene_losses = []
    scene_labels = []
    for match in matches:
        pair_losses = losses[match]  # Shape: [2, num_contexts, max_length]
        pair_labels = labels[match]  # Shape: [2] 
        
        # Add both possible arrangements to get pairwise data
        scene_losses.append(pair_losses)  
        scene_labels.append(pair_labels)
    
    return scene_losses, scene_labels

def evaluate_intphys_full(data_path, checkpoint_path, device='cuda', context_lengths=[2,4,6,8,10]):
    """Evaluate full IntPhys dataset exactly as Quentin does"""
    
    logger.info(f"Starting full IntPhys evaluation")
    
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
        
        # Extract losses
        scene_losses, scene_labels = extract_losses_for_scene(
            model, scene_data, context_lengths, 
            frames_per_clip=16, stride=2, use_bfloat16=True, device=device
        )
        
        # scene_losses is now a list of [pair_losses] where each pair_losses is [2, num_contexts, max_length]
        # scene_labels is now a list of [pair_labels] where each pair_labels is [2]
        
        all_losses.extend(scene_losses)
        all_labels.extend(scene_labels) 
        all_ids.append(torch.tensor([scene_idx]))
    
    # Now we need to reformat for compute_metrics exactly as Quentin does
    # The original expects losses shaped as [num_pairs * 2, num_contexts, max_length] 
    # and labels as [num_pairs * 2] where pairs are arranged consecutively
    
    lengths = []
    for pair_losses in all_losses:
        lengths.append(pair_losses.size(-1))
    max_length = max(lengths)
    
    # Pad and concatenate all pair losses
    formatted_losses = []
    formatted_labels = []
    
    for pair_losses, pair_labels in zip(all_losses, all_labels):
        # pair_losses: [2, num_contexts, temporal_windows]
        # pair_labels: [2]
        
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
    
    # Compute results for each context length using original compute_metrics
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

def evaluate_intphys_scene(scene_name, data_path, checkpoint_path, device='cuda', context_lengths=[2,4,6,8,10]):
    """Evaluate a single IntPhys scene exactly as Quentin does"""
    
    logger.info(f"Evaluating scene: {scene_name}")
    
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
    
    # Find the scene index
    scene_idx = None
    for i, scene in enumerate(dataset.scenes):
        if scene == scene_name:
            scene_idx = i
            break
    
    if scene_idx is None:
        raise ValueError(f"Scene {scene_name} not found in dataset")
    
    # Get scene data
    scene_data = dataset[scene_idx]
    
    # Extract losses
    scene_losses, scene_labels = extract_losses_for_scene(
        model, scene_data, context_lengths, 
        frames_per_clip=16, stride=2, use_bfloat16=True, device=device
    )
    
    # Compute mean losses for each context length
    results = {}
    for i, context_len in enumerate(context_lengths):
        context_losses = []
        context_labels = []
        for losses, labels in zip(scene_losses, scene_labels):
            context_losses.append(losses[:, i, :])
            context_labels.append(labels)
        
        # Concatenate all pairs
        all_losses = torch.cat(context_losses, dim=0)
        all_labels = torch.cat(context_labels, dim=0)
        
        # Compute mean loss (excluding padding zeros)
        valid_mask = all_losses > 0  # Quentin's losses are never exactly zero
        mean_loss = all_losses[valid_mask].mean().item()
        
        results[f'context_{context_len}'] = {
            'mean_loss': mean_loss,
            'losses': all_losses,
            'labels': all_labels
        }
        
        logger.info(f"Context {context_len}: Mean loss = {mean_loss:.6f}")
    
    return results

if __name__ == "__main__":
    # Full evaluation on O1 dataset
    data_path = "/home/yjianhao/project/video_guidance/dev/O1"  # Point directly to O1 folder
    checkpoint_path = "/home/yjianhao/project/quentinecode/vjepa2/vit-h-open/vith.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Full evaluation with all context lengths as in Quentin's config
    results = evaluate_intphys_full(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        device=device,
        context_lengths=[2, 4, 6, 8, 10]  # Full range as in original config
    )
    
    print(f"\n" + "="*80)
    print("FINAL RESULTS - IntPhys O1 Evaluation")
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
    output_file = "intphys_o1_results_clean_corrected.json"
    save_results = {}
    for context, result in results.items():
        save_results[context] = result  # Save all metrics
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}") 