#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import load_video
from PIL import Image
from datetime import datetime

# Import the IntPhys dataloader
from intphys_dataloader import create_intphys_dataloader
from compute_vjepa_score import get_sliding_window_score_based


class IntPhysEvaluator:
    """Evaluator for IntPhys dataset using V-JEPA."""
    
    def __init__(self, model_name="facebook/vjepa2-vith-fpc64-256", device="auto"):
        """Initialize the evaluator with V-JEPA model."""
        print(f"Loading V-JEPA model: {model_name}")
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="sdpa"
        )
        print("Model loaded successfully!")
    
    def frames_to_video_format(self, clip_tensor):
        """Convert tensor frames to video format for V-JEPA."""
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
        """Compute surprise score for a single clip."""

        # Convert tensor to video format
        video_frames = self.frames_to_video_format(clip_tensor)
        
        # Adjust parameters if video is too short
        if len(video_frames) < kernel_size:
            kernel_size = len(video_frames)
            context_window_size = min(context_window_size, kernel_size - 2)
        
        if context_window_size >= kernel_size:
            context_window_size = max(1, kernel_size // 2)
        
        # Compute surprise using the user's method
        score, loss_arr = get_sliding_window_score_based(
            video_frames, 
            self.model, 
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
        """Compute metrics exactly like the original paper.
        
        Args:
            losses_2d: (num_scenes, 4) tensor of losses
            labels_2d: (num_scenes, 4) tensor of labels  
        """
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
            output_dir = f"./intphys_results_{parent_dir}_{dataset_name}_f{frames_per_clip}_s{frame_step}_{timestamp}"
        
        print(f"Evaluating dataset at: {data_path}")
        print(f"Results will be saved to: {output_dir}")
        
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
        print("EVALUATION RESULTS (Like Original Paper)")
        print("="*100)
        print(f"Dataset: {os.path.basename(data_path.rstrip('/'))}")
        print(f"Number of scenes: {len(all_results)}")
        print(f"Frame skip: {frame_step}")
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
            csv_file = os.path.join(output_dir, "performance.csv")
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
            losses_file = os.path.join(output_dir, "raw_losses.pth")
            torch.save({
                "losses": torch.tensor(all_losses),
                "labels": torch.tensor(all_labels),
                "context_lengths": context_lengths,
                "frame_step": frame_step,
                "dataset": os.path.basename(data_path.rstrip('/'))
            }, losses_file)
            
            # Save detailed metrics
            with open(os.path.join(output_dir, "detailed_metrics.json"), 'w') as f:
                json.dump(context_results, f, indent=2)
            
            print(f"\nResults saved to: {output_dir}")
            print(f"CSV format: {csv_file}")
            print(f"Raw data: {losses_file}")
        
        return context_results, all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA on IntPhys dataset")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to IntPhys dataset (e.g., /path/to/IntPhys/dev/O1/)")
    parser.add_argument("--model_name", type=str, default="facebook/vjepa2-vith-fpc64-256",
                       help="V-JEPA model name")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--frames_per_clip", type=int, default=16,
                       help="Number of frames per clip")
    parser.add_argument("--frame_step", type=int, default=4,
                       help="Frame sampling step")
    parser.add_argument("--output_dir", type=str, default="./intphys_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for model (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Validate data path
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    # Create evaluator
    evaluator = IntPhysEvaluator(model_name=args.model_name, device=args.device)
    
    # Run evaluation
    context_results, all_results = evaluator.evaluate_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        output_dir=args.output_dir if args.output_dir != "./intphys_results" else None
    )
    
    return context_results, all_results


if __name__ == "__main__":
    # Example usage with default paths
    if len(os.sys.argv) == 1:
        # Default run for testing
        data_path = "/home/yjianhao/project/video_guidance/dev/O3"
        
        if os.path.exists(data_path):
            print("Running with default path...")
            evaluator = IntPhysEvaluator()
            context_results, all_results = evaluator.evaluate_dataset(
                data_path=data_path,
                frames_per_clip=8,
                frame_step=2
                # output_dir will be auto-generated
            )
        else:
            print("Default data path not found. Please provide --data_path argument.")
            print("Example usage:")
            print("python reproduce_intphys.py --data_path /path/to/IntPhys/dev/O1/")
    else:
        main()