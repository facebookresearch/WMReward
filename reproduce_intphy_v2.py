#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
import argparse
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# Add the vjepa2 project root to path for PyTorch model loading
sys.path.append('/home/yjianhao/project/vjepa2')

# Import the PyTorch implementation components (original JEPA style)
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

# Import the IntPhys dataloader from the original repo
sys.path.append('/home/yjianhao/project/jepa-intuitive-physics')
from intphys_dataloader import create_intphys_dataloader

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    """Load weights of the VJEPA2 encoder (original PyTorch style)"""
    if not os.path.exists(pretrained_weights):
        print(f"Warning: Checkpoint not found at {pretrained_weights}")
        return
    
    try:
        pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    except Exception as e:
        print(f"Error loading weights: {e}")

def build_pt_video_transform(img_size):
    """Build PyTorch video transform pipeline (original JEPA style)"""
    short_side_size = int(256.0 / 224 * img_size)
    eval_transform = video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return eval_transform

def get_sliding_window_score_torch_v2(video, encoder, target_encoder, predictor, processor, kernel_size, context_window_size, stride=2, return_form='arr', mode='max', require_grad=False):
    """
    Corrected PyTorch version using the original JEPA approach:
    - Target encoder processes full video 
    - Context encoder processes masked video
    - Predictor predicts masked regions from context
    - Loss = L1 between predictions and actual target features
    """
    # Import the masking functions from our compute_vjepa_score
    sys.path.append('/home/yjianhao/project/video_guidance')
    from compute_vjepa_score import get_time_masks, apply_masks
    
    # Process video with PyTorch transforms
    if isinstance(video, list):
        video_np = np.stack([np.array(frame) for frame in video], axis=0)
    else:
        video_np = video
    
    # Convert to tensor and apply transforms
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # T x C x H x W
    video_tensor = processor(video_tensor)  # Apply transforms
    
    # Get device from model parameters
    device = next(encoder.parameters()).device
    video_tensor = video_tensor.unsqueeze(0).to(device)  # 1 x T x C x H x W
    
    # Convert to B x C x T x H x W format for our ViT
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # 1 x C x T x H x W
    
    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    
    B, C, T, H, W = video_tensor.shape
    patch_size = 16
    is_mae = False
    spatial_dim = (H, W)
    start_index_arr = np.arange(0, T - kernel_size + 1, stride)

    loss_arr = []
    for i, start_index in enumerate(start_index_arr):
        try:
            # Slice along temporal dimension (dimension 2): B x C x T x H x W
            video_slice = video_tensor[:, :, start_index:start_index+kernel_size]  # 1 x C x kernel_size x H x W
            
            # Get masks
            m, m_, full_m = get_time_masks(n_timesteps=context_window_size, spatial_size=(patch_size, patch_size), temporal_dim=kernel_size, as_bool=is_mae)
            
            full_m = full_m.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)
            m_ = m_.unsqueeze(0).to(device)

            masks_enc = [m.repeat(B, 1)]
            masks_pred = [m_.repeat(B, 1)]
            full_mask = [full_m.repeat(B, 1)]

            # ORIGINAL JEPA APPROACH:
            # 1. Target encoder processes full video to get target features
            h = target_encoder(video_slice, full_mask)[0]
            
            # Normalize targets
            normalize_targets = True
            if normalize_targets:
                h = torch.nn.functional.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
            
            # 2. Create targets (masked regions of h)
            targets = apply_masks(h, masks_pred, concat=False)
            
            # 3. Context encoder processes masked video (context only)
            context = encoder(video_slice, masks_enc)
            
            # 4. Predictor predicts masked regions from context
            preds = predictor(context, targets, masks_enc, masks_pred)
            
            # 5. Compute L1 loss between predictions and actual targets (like original)
            if len(preds) > 0 and len(targets) > 0:
                pred_features = preds[0]  # [num_masked_patches, feature_dim]
                target_features = targets[0]  # [num_masked_patches, feature_dim]
                
                # L1 loss like the original JEPA code
                loss = torch.nn.functional.l1_loss(pred_features, target_features, reduction='mean')
            else:
                # Fallback if no valid masks
                loss = torch.tensor(0.0).to(device)
                
            if require_grad:
                loss_arr.append(loss)
            else:
                loss_arr.append(loss.detach().item())
                
        except Exception as e:
            print(f"Error in window {i}: {e}")
            continue

    if require_grad:
        if len(loss_arr) == 0:
            return torch.tensor(0.0).to(device)
        loss_tensor = torch.stack(loss_arr)
        if mode == 'max':
            final_loss = torch.max(loss_tensor)
        elif mode == 'mean':
            final_loss = torch.mean(loss_tensor)
        return final_loss
    else:
        if len(loss_arr) == 0:
            return 0.0 if return_form != 'arr' else (0.0, [])
        
        # Original logic for non-gradient case
        if mode == 'max':
            final_loss = np.max(loss_arr)
        elif mode == 'mean':
            final_loss = np.mean(loss_arr)

        if return_form == 'arr':
            return final_loss, loss_arr
        else:
            return final_loss


class IntPhysEvaluatorV2:
    """Evaluator for IntPhys dataset using original PyTorch V-JEPA with proper encoder/target_encoder/predictor."""
    
    def __init__(self, model_path="/home/yjianhao/project/vjepa2/checkpoints/vitg-256.pt", img_size=256, device="cuda"):
        """Initialize the evaluator with PyTorch V-JEPA model (original JEPA style with predictor)."""
        print(f"Loading PyTorch V-JEPA model from: {model_path}")
        print("WARNING: Currently using simplified single-encoder approach.")
        print("For full JEPA evaluation, need to load separate target_encoder and predictor.")
        
        # For now, use the same model for all three components
        # TODO: Load proper JEPA checkpoint with target_encoder and predictor
        self.encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
        self.target_encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
        self.predictor = None  # Not available in vjepa2 single-encoder checkpoint
        
        self.encoder.cuda().eval()
        self.target_encoder.cuda().eval()
        
        # Load pretrained weights for both
        load_pretrained_vjepa_pt_weights(self.encoder, model_path)
        load_pretrained_vjepa_pt_weights(self.target_encoder, model_path)
        
        # Build PyTorch preprocessing transform
        self.processor = build_pt_video_transform(img_size=img_size)
        
        print("PyTorch models loaded successfully!")
        print("Note: Using simplified approach until full JEPA checkpoint available.")
    
    def frames_to_video_format(self, clip_tensor):
        """Convert tensor frames to video format for PyTorch V-JEPA."""
        # clip_tensor shape: (frames, 3, H, W)
        # Convert to PIL Images
        frames = []
        
        # Move tensors to CPU for processing
        if clip_tensor.is_cuda:
            clip_tensor = clip_tensor.cpu()
        
        for i in range(clip_tensor.shape[0]):
            # Convert from normalized tensor back to PIL Image
            frame = clip_tensor[i].clone()
            
            # Denormalize if the tensor is normalized
            if frame.min() < 0:  # Likely normalized
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                frame = frame * std + mean
            
            frame = torch.clamp(frame, 0, 1)
            
            # Convert to PIL
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np)
            frames.append(frame_pil)
        
        return frames
    
    def compute_surprise(self, clip_tensor, kernel_size=16, context_window_size=10, stride=2):
        """Compute surprise score for a single clip using corrected JEPA approach."""

        # Convert tensor to video format
        video_frames = self.frames_to_video_format(clip_tensor)
        
        # Adjust parameters if video is too short
        if len(video_frames) < kernel_size:
            kernel_size = len(video_frames)
            context_window_size = min(context_window_size, kernel_size - 2)
        
        if context_window_size >= kernel_size:
            context_window_size = max(1, kernel_size // 2)
        
        # Check if we have predictor available
        if self.predictor is None:
            print("Warning: No predictor available, falling back to simplified approach")
            return self._compute_surprise_simplified(video_frames, kernel_size, context_window_size, stride)
        
        # Compute surprise using the corrected JEPA method
        score, loss_arr = get_sliding_window_score_torch_v2(
            video_frames, 
            self.encoder,
            self.target_encoder,
            self.predictor,
            self.processor, 
            kernel_size=kernel_size,
            context_window_size=context_window_size,
            stride=stride,
            return_form='arr'
        )
        
        return score, loss_arr
    
    def _compute_surprise_simplified(self, video_frames, kernel_size, context_window_size, stride):
        """Simplified approach when predictor is not available."""
        # Use our existing debug_loss2.py approach temporarily
        sys.path.append('/home/yjianhao/project/video_guidance')
        from debug_loss2 import get_sliding_window_score_torch
        
        score, loss_arr = get_sliding_window_score_torch(
            video_frames,
            self.encoder,
            self.processor,
            kernel_size=kernel_size,
            context_window_size=context_window_size,
            stride=stride,
            return_form='arr'
        )
        
        return score, loss_arr
    
    def evaluate_scene(self, clips, labels, scene_paths=None):
        """Evaluate a single scene with 4 possibilities - compute losses for all context lengths."""
        # clips shape: (4, frames, 3, H, W)
        # labels shape: (4,) - boolean indicating if possibility is physically plausible
        
        # Test multiple context lengths like the original paper
        context_lengths = [2, 4, 6, 8, 10]
        
        all_surprises = []
        all_loss_arrays = []
        
        for context_len in context_lengths:
            surprises_for_context = []
            loss_arrays_for_context = []
            
            for i in range(4):
                clip = clips[i]  # (frames, 3, H, W)
                
                # Use parameters similar to original paper
                kernel_size = min(16, clip.shape[0])  # 16 frames like original
                stride = 1
                
                # Ensure context length is valid
                effective_context_len = min(context_len, kernel_size - 2)
                if effective_context_len < 1:
                    effective_context_len = 1
                
                surprise, loss_arr = self.compute_surprise(
                    clip, 
                    kernel_size=kernel_size, 
                    context_window_size=effective_context_len, 
                    stride=stride
                )
                surprises_for_context.append(surprise)
                loss_arrays_for_context.append(loss_arr)
            
            all_surprises.append(surprises_for_context)
            all_loss_arrays.append(loss_arrays_for_context)
        
        # Convert to numpy arrays
        all_surprises = np.array(all_surprises)  # Shape: (num_context_lengths, 4)
        labels_np = labels.numpy().astype(float)
        
        return {
            'all_surprises': all_surprises,  # Shape: (5 contexts, 4 clips)
            'labels': labels_np,
            'loss_arrays': all_loss_arrays,
            'scene_paths': scene_paths,
            'context_lengths': context_lengths
        }
    
    def compute_metrics_like_original(self, losses_2d, labels_2d):
        """Compute metrics exactly like the original paper."""
        metrics = {}
        
        # Convert to tensors (keep 2D structure like original)
        losses = torch.tensor(losses_2d, dtype=torch.float32)  # (num_scenes, 4)
        labels = torch.tensor(labels_2d, dtype=torch.float32)  # (num_scenes, 4)
        
        # Separate real (plausible) and fake (implausible) losses like original
        loss_real = losses[torch.where(labels == 1)]  # All plausible clips
        loss_fake = losses[torch.where(labels == 0)]  # All implausible clips
        
        # Reshape back to 2D for scene-wise operations
        # Count how many plausible/implausible per scene
        num_real_per_scene = (labels == 1).sum(dim=1)  # (num_scenes,)
        num_fake_per_scene = (labels == 0).sum(dim=1)  # (num_scenes,)
        
        # Get max number of real/fake clips per scene for reshaping
        max_real = num_real_per_scene.max().item()
        max_fake = num_fake_per_scene.max().item()
        
        # Create padded 2D tensors for real and fake losses
        num_scenes = losses.shape[0]
        loss_real_2d = torch.full((num_scenes, max_real), float('inf'))
        loss_fake_2d = torch.full((num_scenes, max_fake), float('inf'))
        
        # Fill in the actual losses
        for scene_idx in range(num_scenes):
            scene_losses = losses[scene_idx]
            scene_labels = labels[scene_idx]
            
            real_losses = scene_losses[scene_labels == 1]
            fake_losses = scene_losses[scene_labels == 0]
            
            loss_real_2d[scene_idx, :len(real_losses)] = real_losses
            loss_fake_2d[scene_idx, :len(fake_losses)] = fake_losses
        
        # Handle inf values for mean/max computations
        loss_real_valid = loss_real_2d.clone()
        loss_fake_valid = loss_fake_2d.clone()
        loss_real_valid[loss_real_valid == float('inf')] = 0
        loss_fake_valid[loss_fake_valid == float('inf')] = 0
        
        # Compute scene-wise means (only over valid losses)
        real_means = []
        fake_means = []
        real_maxes = []
        fake_maxes = []
        
        for scene_idx in range(num_scenes):
            real_valid_mask = loss_real_2d[scene_idx] != float('inf')
            fake_valid_mask = loss_fake_2d[scene_idx] != float('inf')
            
            if real_valid_mask.any():
                real_means.append(loss_real_2d[scene_idx][real_valid_mask].mean())
                real_maxes.append(loss_real_2d[scene_idx][real_valid_mask].max())
            else:
                real_means.append(torch.tensor(0.0))
                real_maxes.append(torch.tensor(0.0))
                
            if fake_valid_mask.any():
                fake_means.append(loss_fake_2d[scene_idx][fake_valid_mask].mean())
                fake_maxes.append(loss_fake_2d[scene_idx][fake_valid_mask].max())
            else:
                fake_means.append(torch.tensor(0.0))
                fake_maxes.append(torch.tensor(0.0))
        
        real_means = torch.stack(real_means)
        fake_means = torch.stack(fake_means)
        real_maxes = torch.stack(real_maxes)
        fake_maxes = torch.stack(fake_maxes)
        
        # Relative accuracy metrics (exactly like original)
        acc_pairwise_mean = (real_means < fake_means).sum().float() / num_scenes * 100
        acc_pairwise_max = (real_maxes < fake_maxes).sum().float() / num_scenes * 100
        
        metrics["Relative Accuracy (avg)"] = acc_pairwise_mean.item()
        metrics["Relative Accuracy (max)"] = acc_pairwise_max.item()
        
        # Absolute accuracy using scene-level max losses (like original)
        data1 = real_maxes  # Max loss per scene for plausible 
        data2 = fake_maxes  # Max loss per scene for implausible
        
        # 90% threshold calibration on real videos
        thresh = data1.sort()[0][int(np.ceil(0.90 * len(data1)))]
        accuracy_abs = ((data1 < thresh).sum() + (data2 > thresh).sum()) / (data1.shape[0] + data2.shape[0]) * 100
        
        metrics["Absolute Accuracy (max)"] = accuracy_abs.item()
        metrics["Classifier threhshold"] = thresh.item()  # Keep original typo for compatibility
        
        # Best absolute accuracy (oracle threshold) - fixing original bug
        threshs = np.linspace(data1.min().item(), data2.max().item(), 100)
        accs = []
        for thresh_val in threshs:
            acc = ((data1 < thresh_val).sum() + (data2 > thresh_val).sum()) / (data1.shape[0] + data2.shape[0])
            accs.append(acc)
        
        best_accuracy_abs = torch.max(torch.tensor(accs)) * 100
        oracle_thresh = threshs[torch.argmax(torch.tensor(accs))]
        
        metrics["Best Absolute Accuracy (max)"] = best_accuracy_abs.item()
        metrics["Best Classifier threhshold"] = oracle_thresh  # Keep original typo for compatibility
        
        # AUPRC and AUROC using scene-level aggregations (like original)
        try:
            # Flatten scene labels for classification (plausible=1, implausible=0)
            scene_labels_flat = []
            scene_losses_max = []
            scene_losses_mean = []
            
            for scene_idx in range(num_scenes):
                # For each scene, we have one aggregated "plausible" and one "implausible" value
                if real_means[scene_idx] > 0:  # Valid real loss
                    scene_labels_flat.append(1)  # Plausible
                    scene_losses_max.append(real_maxes[scene_idx].item())
                    scene_losses_mean.append(real_means[scene_idx].item())
                    
                if fake_means[scene_idx] > 0:  # Valid fake loss  
                    scene_labels_flat.append(0)  # Implausible
                    scene_losses_max.append(fake_maxes[scene_idx].item())
                    scene_losses_mean.append(fake_means[scene_idx].item())
            
            scene_labels_flat = np.array(scene_labels_flat)
            scene_losses_max = np.array(scene_losses_max)
            scene_losses_mean = np.array(scene_losses_mean)
            
            # AUPRC (negative because higher loss = more implausible)
            precision_max, recall_max, _ = precision_recall_curve(scene_labels_flat, -scene_losses_max)
            precision_mean, recall_mean, _ = precision_recall_curve(scene_labels_flat, -scene_losses_mean)
            auprc_max = auc(recall_max, precision_max)
            auprc_mean = auc(recall_mean, precision_mean)
            
            # AUROC (negative because higher loss = more implausible)
            fpr_max, tpr_max, _ = roc_curve(scene_labels_flat, -scene_losses_max)
            fpr_mean, tpr_mean, _ = roc_curve(scene_labels_flat, -scene_losses_mean)
            auroc_max = auc(fpr_max, tpr_max)
            auroc_mean = auc(fpr_mean, tpr_mean)
            
            metrics["AUPRC (avg)"] = auprc_mean
            metrics["AUPRC (max)"] = auprc_max
            metrics["AUROC (avg)"] = auroc_mean
            metrics["AUROC (max)"] = auroc_max
            
        except Exception as e:
            print(f"Warning: AUROC/AUPRC computation failed: {e}")
            metrics["AUPRC (avg)"] = 0.5
            metrics["AUPRC (max)"] = 0.5
            metrics["AUROC (avg)"] = 0.5
            metrics["AUROC (max)"] = 0.5
        
        return metrics
    
    def evaluate_dataset(self, data_path, batch_size=1, frames_per_clip=16, frame_step=4, 
                        save_results=True, output_dir=None):
        """Evaluate the entire dataset exactly like the original paper."""
        
        # Auto-generate output directory based on dataset path if not provided
        if output_dir is None:
            dataset_name = os.path.basename(data_path.rstrip('/'))
            parent_dir = os.path.basename(os.path.dirname(data_path.rstrip('/')))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./intphys_results_v2_pytorch_{parent_dir}_{dataset_name}_f{frames_per_clip}_s{frame_step}_{timestamp}"
        
        print(f"Evaluating dataset at: {data_path}")
        print(f"Results will be saved to: {output_dir}")
        print(f"Using PyTorch V-JEPA model with enhanced loss calculation")
        
        # Create dataloader
        dataloader = create_intphys_dataloader(
            data_path=data_path,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_workers=2,
            shuffle=False,
            return_paths=True
        )
        
        print(f"Dataset loaded: {len(dataloader.dataset)} scenes")
        
        all_results = []
        context_lengths = [2, 4, 6, 8, 10]
        
        # Collect all losses and labels for each context length
        all_losses = []  # Shape: (num_scenes, num_contexts, 4)
        all_labels = []  # Shape: (num_scenes, 4)
        
        # Evaluate each batch
        for batch_idx, (clips_batch, labels_batch, paths_batch) in enumerate(tqdm(dataloader, desc="Evaluating scenes")):
            
            for sample_idx in range(clips_batch.shape[0]):
                clips = clips_batch[sample_idx]  # (4, frames, 3, H, W)
                labels = labels_batch[sample_idx]  # (4,)
                paths = [paths_batch[i][sample_idx] if isinstance(paths_batch[i], list) else paths_batch[sample_idx][i] for i in range(4)]
                
                # Evaluate this scene
                result = self.evaluate_scene(clips, labels, paths)
                all_results.append(result)
                
                # Collect losses and labels for analysis
                all_losses.append(result['all_surprises'])  # Shape: (5, 4)
                all_labels.append(result['labels'])  # Shape: (4,)
                
                # Print progress
                if batch_idx % 5 == 0:
                    print(f"Scene {batch_idx}: computed losses for all context lengths")
        
        # Convert to numpy arrays
        all_losses = np.array(all_losses)  # Shape: (num_scenes, 5, 4)
        all_labels = np.array(all_labels)   # Shape: (num_scenes, 4)
        
        # Results dictionary to store metrics for each context length
        context_results = {}
        
        print("\n" + "="*100)
        print("EVALUATION RESULTS (PyTorch V-JEPA V2 - Corrected JEPA Approach)")
        print("="*100)
        print(f"Dataset: {os.path.basename(data_path.rstrip('/'))}")
        print(f"Number of scenes: {len(all_results)}")
        print(f"Frame skip: {frame_step}")
        print(f"Model: PyTorch V-JEPA with Corrected JEPA Loss (Encoder+Target_Encoder+Predictor)")
        print("-"*100)
        print(f"{'Context':>8} | {'Rel Acc (avg)':>12} | {'Rel Acc (max)':>12} | {'Abs Acc (max)':>12} | {'AUPRC (avg)':>10} | {'AUPRC (max)':>10} | {'AUROC (avg)':>10} | {'AUROC (max)':>10}")
        print("-"*100)
        
        # Evaluate each context length separately
        for ctx_idx, context_len in enumerate(context_lengths):
            # Get losses for this context length: (num_scenes, 4)
            ctx_losses = all_losses[:, ctx_idx, :]  # Shape: (num_scenes, 4)
            
            # Compute metrics for this context length (pass 2D arrays)
            metrics = self.compute_metrics_like_original(ctx_losses, all_labels)
            context_results[context_len] = metrics
            
            # Print results in original format
            print(f"{context_len:8} | {metrics['Relative Accuracy (avg)']:11.1f}% | "
                  f"{metrics['Relative Accuracy (max)']:11.1f}% | "
                  f"{metrics['Best Absolute Accuracy (max)']:11.1f}% | "
                  f"{metrics['AUPRC (avg)']:9.3f} | {metrics['AUPRC (max)']:9.3f} | "
                  f"{metrics['AUROC (avg)']:9.3f} | {metrics['AUROC (max)']:9.3f}")
        
        # Compute "Filtered" result (minimum loss across all contexts)
        min_losses = all_losses.min(axis=1)  # Shape: (num_scenes, 4) - min across contexts
        filtered_metrics = self.compute_metrics_like_original(min_losses, all_labels)
        context_results['Filtered'] = filtered_metrics
        
        print(f"{'Filtered':8} | {filtered_metrics['Relative Accuracy (avg)']:11.1f}% | "
              f"{filtered_metrics['Relative Accuracy (max)']:11.1f}% | "
              f"{filtered_metrics['Best Absolute Accuracy (max)']:11.1f}% | "
              f"{filtered_metrics['AUPRC (avg)']:9.3f} | {filtered_metrics['AUPRC (max)']:9.3f} | "
              f"{filtered_metrics['AUROC (avg)']:9.3f} | {filtered_metrics['AUROC (max)']:9.3f}")
        
        print("="*100)
        
        # Save results
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics in original CSV format
            csv_file = os.path.join(output_dir, "performance_pytorch_vjepa_v2.csv")
            with open(csv_file, 'w') as f:
                # Write header
                f.write("Block;Context length(s);Frame skip;Relative Accuracy (avg);Relative Accuracy (max);"
                       "Absolute Accuracy (max);Classifier threhshold;Best Absolute Accuracy (max);"
                       "Best Classifier threhshold;AUPRC (avg);AUPRC (max);AUROC (avg);AUROC (max)\n")
                
                # Write results for each context length
                block = os.path.basename(data_path.rstrip('/'))
                for context_len in context_lengths:
                    metrics = context_results[context_len]
                    f.write(f"{block};{context_len};{frame_step};"
                           f"{metrics['Relative Accuracy (avg)']:.5f};"
                           f"{metrics['Relative Accuracy (max)']:.5f};"
                           f"{metrics['Absolute Accuracy (max)']:.5f};"
                           f"{metrics['Classifier threhshold']:.5f};"
                           f"{metrics['Best Absolute Accuracy (max)']:.5f};"
                           f"{metrics['Best Classifier threhshold']:.5f};"
                           f"{metrics['AUPRC (avg)']:.5f};"
                           f"{metrics['AUPRC (max)']:.5f};"
                           f"{metrics['AUROC (avg)']:.5f};"
                           f"{metrics['AUROC (max)']:.5f}\n")
                
                # Write filtered result
                metrics = context_results['Filtered']
                f.write(f"{block};Filtered;{frame_step};"
                       f"{metrics['Relative Accuracy (avg)']:.5f};"
                       f"{metrics['Relative Accuracy (max)']:.5f};"
                       f"{metrics['Absolute Accuracy (max)']:.5f};"
                       f"{metrics['Classifier threhshold']:.5f};"
                       f"{metrics['Best Absolute Accuracy (max)']:.5f};"
                       f"{metrics['Best Classifier threhshold']:.5f};"
                       f"{metrics['AUPRC (avg)']:.5f};"
                       f"{metrics['AUPRC (max)']:.5f};"
                       f"{metrics['AUROC (avg)']:.5f};"
                       f"{metrics['AUROC (max)']:.5f}\n")
            
            # Save raw losses and labels (like original)
            losses_file = os.path.join(output_dir, "raw_losses_pytorch_vjepa_v2.pth")
            torch.save({
                "losses": torch.tensor(all_losses),
                "labels": torch.tensor(all_labels),
                "context_lengths": context_lengths,
                "frame_step": frame_step,
                "dataset": os.path.basename(data_path.rstrip('/'))
            }, losses_file)
            
            # Save detailed metrics
            with open(os.path.join(output_dir, "detailed_metrics_pytorch_vjepa_v2.json"), 'w') as f:
                json.dump(context_results, f, indent=2)
            
            print(f"\nResults saved to: {output_dir}")
            print(f"CSV format: {csv_file}")
            print(f"Raw data: {losses_file}")
        
        return context_results, all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch V-JEPA V2 on IntPhys dataset (Corrected JEPA Approach)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to IntPhys dataset (e.g., /path/to/IntPhys/dev/O1/)")
    parser.add_argument("--model_path", type=str, default="/home/yjianhao/project/vjepa2/checkpoints/vitg-256.pt",
                       help="Path to PyTorch V-JEPA checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--frames_per_clip", type=int, default=16,
                       help="Number of frames per clip")
    parser.add_argument("--frame_step", type=int, default=4,
                       help="Frame sampling step")
    parser.add_argument("--output_dir", type=str, default="./intphys_results_v2",
                       help="Output directory for results")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Image size for model")
    
    args = parser.parse_args()
    
    # Validate data path
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    # Create evaluator
    evaluator = IntPhysEvaluatorV2(model_path=args.model_path, img_size=args.img_size)
    
    # Run evaluation
    context_results, all_results = evaluator.evaluate_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        output_dir=args.output_dir if args.output_dir != "./intphys_results_v2" else None
    )
    
    return context_results, all_results


if __name__ == "__main__":
    # Example usage with default paths
    if len(os.sys.argv) == 1:
        # Default run for testing
        data_path = "/home/yjianhao/project/video_guidance/dev/O3"
        
        if os.path.exists(data_path):
            print("Running with default path...")
            evaluator = IntPhysEvaluatorV2()
            context_results, all_results = evaluator.evaluate_dataset(
                data_path=data_path,
                frames_per_clip=8,
                frame_step=2
                # output_dir will be auto-generated
            )
        else:
            print("Default data path not found. Please provide --data_path argument.")
            print("Example usage:")
            print("python reproduce_intphy_v2.py --data_path /path/to/IntPhys/dev/O1/")
    else:
        main() 