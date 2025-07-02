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
import copy
import torch.nn.functional as F
from einops import rearrange

# Add the quentinecode vjepa2 project root to path for their implementation
sys.path.append('/home/yjianhao/project/quentinecode/vjepa2')

# Import Quentin's JEPA implementation components
from app.vjepa.modelcustom.vit_encoder_predictor_noar_targets import init_module
from app.vjepa.transforms import make_transforms
from evals.intuitive_physics_modelcustom.utils import get_time_masks, get_dataset_paths, PROPERTIES_BY_DATASET
from evals.intuitive_physics_modelcustom.data_manager import init_data

# Import the IntPhys dataloader from the original repo
sys.path.append('/home/yjianhao/project/jepa-intuitive-physics')
from intphys_dataloader import create_intphys_dataloader

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def apply_masks(x, masks, concat=False):
    """Apply masks to features (from Quentin's implementation)"""
    if concat:
        return torch.cat([m.unsqueeze(0) for m in masks], dim=0)
    return [x[i, m] for i, m in enumerate(masks)]


class IntPhysEvaluatorV2:
    """Evaluator for IntPhys dataset using Quentin's full JEPA implementation."""
    
    def __init__(self, model_path="/home/yjianhao/project/quentinecode/vjepa2/vit-h-open/vith.pt", img_size=256, device="cuda"):
        """Initialize the evaluator with Quentin's full JEPA model."""
        print(f"Loading Quentin's full JEPA model from: {model_path}")
        
        # Use Quentin's exact model configuration from config file
        model_kwargs = {
            "resolution": img_size,
            "encoder": {
                "model_name": "vit_huge",
                "checkpoint_key": "encoder",
                "is_causal": False,
                "local_window": [-1, -1, -1],
                "uniform_power": True,
                "use_activation_checkpointing": True,
                "use_mask_tokens": True,
                "use_rope": True,
                "zero_init_mask_tokens": True,
                "num_frames": 16
            },
            "target_encoder": {
                "checkpoint_key": "target_encoder"
            },
            "predictor": {
                "model_name": "vit_predictor",
                "checkpoint_key": "predictor",
                "depth": 12,
                "is_causal": False,
                "local_window": [-1, -1, -1],
                "num_heads": 12,
                "uniform_power": True,
                "use_activation_checkpointing": True,
                "use_mask_tokens": True,
                "use_rope": True,
                "zero_init_mask_tokens": True,
                "num_mask_tokens": 10,
                "num_frames": 16
            }
        }
        
        wrapper_kwargs = {
            "no_predictor": False
        }
        
        # Initialize using Quentin's model wrapper exactly like original
        self.model = init_module(
            frames_per_clip=16,
            nb_context_frames=1,  # Use 1 like original, will be updated dynamically
            checkpoint=model_path,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs
        )
        
        self.model.to(device).eval()
        
        # Build transforms using Quentin's approach
        self.transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=[1/1, 1/1],
            random_resize_scale=[1.0, 1.0],
            reprob=0.,
            auto_augment=False,
            motion_shift=False,
            crop_size=img_size
        )
        
        self.device = device
        self.img_size = img_size
        
        print("Quentin's full JEPA model loaded successfully!")
    
    def compute_surprise_quentin_style(self, clips, context_lengths, stride=2):
        """Compute surprise using Quentin's exact JEPA methodology."""
        # clips shape: (4, 3, frames, H, W) after transformation
        clips = clips.to(self.device)
        B, C, T, H, W = clips.shape
        
        # Clips are already in B x C x T x H x W format after transforms
        
        all_losses = []  # Shape: (context_lengths, 4)
        
        for context_len in context_lengths:
            # Update model context length exactly like Quentin
            self.model.nb_context_frames = context_len
            self.model.frames_per_clip = 16  # Keep fixed like Quentin
            try:
                self.model.grid_depth = self.model.frames_per_clip // self.model.encoder.tubelet_size
            except:
                try:
                    self.model.grid_depth = self.model.frames_per_clip // self.model.encoder.backbone.tubelet_size
                except:
                    self.model.grid_depth = self.model.frames_per_clip // 2
            
            # Create sliding windows exactly like Quentin
            pieces = clips.unfold(2, self.model.frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
            pieces = pieces.flatten(0, 1)  # Flatten batch and window dimensions
            pieces = rearrange(pieces, "b t c h w -> b c t h w")
            
            # Process in chunks like Quentin (to handle memory)
            CHUNK_SIZE = 2
            chunked_preds = []
            chunked_targets = []
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                for chunk_id in range(int(np.ceil(pieces.shape[0] / CHUNK_SIZE))):
                    chunk = pieces[CHUNK_SIZE * chunk_id:CHUNK_SIZE * (chunk_id + 1)]
                    
                    if chunk.shape[0] == 0:
                        continue
                    
                    # Use Quentin's model forward pass
                    preds, targets = self.model(chunk)
                    chunked_preds.append(preds.cpu())
                    chunked_targets.append(targets.cpu())
                
                if chunked_preds:
                    preds = torch.vstack(chunked_preds)
                    targets = torch.vstack(chunked_targets)
                    
                    # Reshape back to (num_videos, num_windows, ...)
                    preds = preds.view(B, -1, *preds.shape[1:])
                    targets = targets.view(B, -1, *targets.shape[1:])
                    
                    # Compute L1 loss exactly like Quentin: mean over tokens and features, then handle windows
                    loss = F.l1_loss(preds, targets, reduction="none").mean((2, 3)).detach()

                    print("Loss:",loss)
                    
                    # Handle window aggregation like Quentin - take max across windows for each video
                    if loss.dim() > 1:
                        loss = loss.max(dim=1)[0]  # Max across windows
                    
                    all_losses.append(loss.cpu())
                else:
                    # Fallback if no valid chunks
                    all_losses.append(torch.zeros(B))
        
        # Stack to get shape (context_lengths, 4)
        all_losses = torch.stack(all_losses)  # (num_contexts, 4)
        
        return all_losses
    
    def evaluate_scene(self, clips, labels, scene_paths=None):
        """Evaluate a single scene using Quentin's methodology."""
        # clips shape: (4, frames, 3, H, W)
        # labels shape: (4,) - boolean indicating if possibility is physically plausible
        
        # Use same context lengths as Quentin
        context_lengths = [2, 4, 6, 8, 10]
        
        # Apply transforms like Quentin - clips are already in correct format
        # Convert to (frames, H, W, 3) for transforms
        transformed_clips = []
        for i in range(clips.shape[0]):
            clip = clips[i]  # (frames, 3, H, W)
            # Convert to PIL/numpy format expected by transforms: (frames, H, W, 3)
            clip_np = clip.permute(0, 2, 3, 1).cpu().numpy()  # (frames, H, W, 3)
            # Convert to 0-255 range if needed
            if clip_np.max() <= 1.0:
                clip_np = (clip_np * 255).astype(np.uint8)
            clip_transformed = self.transform(clip_np)  # Returns (3, frames, H, W)
            transformed_clips.append(clip_transformed)
        
        transformed_clips = torch.stack(transformed_clips)  # (4, 3, frames, H, W)
        
        # Compute losses using Quentin's approach
        all_surprises = self.compute_surprise_quentin_style(transformed_clips, context_lengths)
        
        return {
            'all_surprises': all_surprises.numpy(),  # Shape: (5 contexts, 4 clips)
            'labels': labels.numpy().astype(float),
            'scene_paths': scene_paths,
            'context_lengths': context_lengths
        }
    
    def compute_metrics_like_original(self, losses_2d, labels_2d):
        """Compute metrics exactly like the original implementation."""
        metrics = {}
        
        # Convert to tensors matching original format
        losses = torch.tensor(losses_2d, dtype=torch.float32)  # (num_scenes, 4)
        labels = torch.tensor(labels_2d, dtype=torch.float32)  # (num_scenes, 4)
        
        # Extract real and fake losses exactly like original
        loss_real = losses[torch.where(labels == 1)]  # All plausible clips
        loss_fake = losses[torch.where(labels == 0)]  # All implausible clips
        
        # Reshape to match original's structure - group by scenes
        num_scenes = losses.shape[0]
        # Assuming 2 real and 2 fake per scene (typical IntPhys structure)
        loss_real = loss_real.view(num_scenes, -1)  # (num_scenes, num_real_per_scene)
        loss_fake = loss_fake.view(num_scenes, -1)  # (num_scenes, num_fake_per_scene)
        
        # Compute metrics exactly like original implementation
        acc_pairwise_mean = (loss_real.mean(1) < loss_fake.mean(1)).sum() / loss_real.shape[0] * 100
        acc_pairwise_max = (loss_real.max(1)[0] < loss_fake.max(1)[0]).sum() / loss_real.shape[0] * 100
        
        metrics["Relative Accuracy (avg)"] = acc_pairwise_mean.item()
        metrics["Relative Accuracy (max)"] = acc_pairwise_max.item()
        
        # Compute single video classification exactly like original
        data1 = loss_real.max(1)[0]  # Max loss per scene for real videos
        data2 = loss_fake.max(1)[0]  # Max loss per scene for fake videos
        
        # Use threshold = 0 like original (comment shows thresh = 0)
        thresh = 0  # Following original implementation
        accuracy_abs = ((data1 < thresh).sum() + (data2 > thresh).sum()) / (data1.shape[0] + data2.shape[0]) * 100
        
        metrics["Absolute Accuracy (max)"] = accuracy_abs.item()
        metrics["Classifier threhshold"] = thresh
        
        # Best threshold search exactly like original (fixing the bug)
        threshs = np.linspace(data1.min().item(), data2.max().item(), 100)
        accs = []
        for thresh_val in threshs:
            # Fix: use data2.shape[0] instead of data1.shape[0] for second term
            acc = ((data1 < thresh_val).sum() + (data2 > thresh_val).sum()) / (data1.shape[0] + data2.shape[0])
            accs.append(acc)
        
        best_accuracy_abs = torch.max(torch.tensor(accs)) * 100
        oracle_thresh = threshs[torch.argmax(torch.tensor(accs))]
        
        metrics["Best Absolute Accuracy (max)"] = best_accuracy_abs.item()
        metrics["Best Classifier threhshold"] = oracle_thresh
        
        # AUPRC and AUROC exactly like original
        try:
            # Flatten labels for classification
            labels_flat = labels.flatten()
            losses_flat_max = losses.max(1)[0].repeat_interleave(losses.shape[1])
            losses_flat_mean = losses.mean(1).repeat_interleave(losses.shape[1])
            
            # AUPRC (negative because higher loss = more implausible)
            precision_max, recall_max, _ = precision_recall_curve(labels_flat, -losses_flat_max)
            precision_mean, recall_mean, _ = precision_recall_curve(labels_flat, -losses_flat_mean)
            auprc_max = auc(recall_max, precision_max)
            auprc_mean = auc(recall_mean, precision_mean)
            
            # AUROC (negative because higher loss = more implausible)
            fpr_max, tpr_max, _ = roc_curve(labels_flat, -losses_flat_max)
            fpr_mean, tpr_mean, _ = roc_curve(labels_flat, -losses_flat_mean)
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
    
    def evaluate_dataset(self, data_path, batch_size=1, frames_per_clip=16, frame_step=2, 
                        save_results=True, output_dir=None):
        """Evaluate the entire dataset using Quentin's methodology."""
        
        # Auto-generate output directory based on dataset path if not provided
        if output_dir is None:
            dataset_name = os.path.basename(data_path.rstrip('/'))
            parent_dir = os.path.basename(os.path.dirname(data_path.rstrip('/')))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./intphys_results_v2_quentin_style_{parent_dir}_{dataset_name}_f{frames_per_clip}_s{frame_step}_{timestamp}"
        
        print(f"Evaluating dataset at: {data_path}")
        print(f"Results will be saved to: {output_dir}")
        print(f"Using Quentin's full JEPA model (encoder + target_encoder + predictor)")
        
        # Create dataloader with parameters matching original
        dataloader = create_intphys_dataloader(
            data_path=data_path,
            batch_size=batch_size,
            frames_per_clip=99//frame_step,  # Use same frame calculation as original
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
                print("All clips shape: ",clips.shape)
                labels = labels_batch[sample_idx]  # (4,)
                paths = [paths_batch[i][sample_idx] if isinstance(paths_batch[i], list) else paths_batch[sample_idx][i] for i in range(4)]
                
                # Evaluate this scene using Quentin's approach
                result = self.evaluate_scene(clips, labels, paths)
                all_results.append(result)
                
                # Collect losses and labels for analysis
                all_losses.append(result['all_surprises'])  # Shape: (5, 4)
                all_labels.append(result['labels'])  # Shape: (4,)
                
                # Print progress
                if batch_idx % 5 == 0:
                    print(f"Scene {batch_idx}: computed losses using Quentin's JEPA approach")
        
        # Convert to numpy arrays
        all_losses = np.array(all_losses)  # Shape: (num_scenes, 5, 4)
        all_labels = np.array(all_labels)   # Shape: (num_scenes, 4)
        
        # Results dictionary to store metrics for each context length
        context_results = {}
        
        print("\n" + "="*100)
        print("EVALUATION RESULTS (Quentin's Full JEPA Implementation)")
        print("="*100)
        print(f"Dataset: {os.path.basename(data_path.rstrip('/'))}")
        print(f"Number of scenes: {len(all_results)}")
        print(f"Frame skip: {frame_step}")
        print(f"Model: Quentin's Full JEPA (Encoder + Target_Encoder + Predictor)")
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
            csv_file = os.path.join(output_dir, "performance_quentin_style_jepa.csv")
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
            losses_file = os.path.join(output_dir, "raw_losses_quentin_style_jepa.pth")
            torch.save({
                "losses": torch.tensor(all_losses),
                "labels": torch.tensor(all_labels),
                "context_lengths": context_lengths,
                "frame_step": frame_step,
                "dataset": os.path.basename(data_path.rstrip('/'))
            }, losses_file)
            
            # Save detailed metrics
            with open(os.path.join(output_dir, "detailed_metrics_quentin_style_jepa.json"), 'w') as f:
                json.dump(context_results, f, indent=2)
            
            print(f"\nResults saved to: {output_dir}")
            print(f"CSV format: {csv_file}")
            print(f"Raw data: {losses_file}")
        
        return context_results, all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate using Quentin's Full JEPA on IntPhys dataset")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to IntPhys dataset (e.g., /path/to/IntPhys/dev/O1/)")
    parser.add_argument("--model_path", type=str, default="/home/yjianhao/project/quentinecode/vjepa2/vit-h-open/vith.pt",
                       help="Path to Quentin's full JEPA checkpoint")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--frames_per_clip", type=int, default=16,
                       help="Number of frames per clip")
    parser.add_argument("--frame_step", type=int, default=2,
                       help="Frame sampling step")
    parser.add_argument("--output_dir", type=str, default="./intphys_results_quentin_style",
                       help="Output directory for results")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Image size for model")
    
    args = parser.parse_args()
    
    # Validate data path
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    # Create evaluator using Quentin's approach
    evaluator = IntPhysEvaluatorV2(model_path=args.model_path, img_size=args.img_size)
    
    # Run evaluation
    context_results, all_results = evaluator.evaluate_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        output_dir=args.output_dir if args.output_dir != "./intphys_results_quentin_style" else None
    )
    
    return context_results, all_results


if __name__ == "__main__":
    # Example usage with default paths
    if len(os.sys.argv) == 1:
        # Default run for testing
        data_path = "/home/yjianhao/project/video_guidance/dev/O3"
        
        if os.path.exists(data_path):
            print("Running with default path using Quentin's approach...")
            evaluator = IntPhysEvaluatorV2()
            context_results, all_results = evaluator.evaluate_dataset(
                data_path=data_path,
                frames_per_clip=16,
                frame_step=2
                # output_dir will be auto-generated
            )
        else:
            print("Default data path not found. Please provide --data_path argument.")
            print("Example usage:")
            print("python reproduce_intphy_v2.py --data_path /path/to/IntPhys/dev/O1/")
    else:
        main() 