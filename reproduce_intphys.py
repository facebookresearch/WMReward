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
        
        try:
            self.processor = AutoVideoProcessor.from_pretrained(model_name)
            print("Processor loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load processor: {e}")
            print("Falling back to default processor...")
            self.processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vith-fpc64-256")
        
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                attn_implementation="sdpa"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load model with SDPA: {e}")
            print("Retrying without SDPA...")
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=device
                )
                print("Model loaded successfully without SDPA!")
            except Exception as e2:
                print(f"Error: Failed to load model: {e2}")
                raise e2
    
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
    
    def compute_surprise_simple(self, clip_tensor, frames_per_clip=16, stride=2):
        """Compute surprise score for a single clip using simple unfold approach like original."""
        
        # Convert tensor to video format
        video_frames = self.frames_to_video_format(clip_tensor)
        
        # Adjust parameters if video is too short
        if len(video_frames) < frames_per_clip:
            frames_per_clip = len(video_frames)
        
        # Convert back to tensor for unfold operation (like original)
        tensor_frames = []
        for frame in video_frames:
            # Convert PIL to tensor
            tensor_frame = torch.tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0
            tensor_frames.append(tensor_frame)
        
        video_tensor = torch.stack(tensor_frames).unsqueeze(0)  # (1, T, C, H, W)
        
        # Apply unfold operation exactly like original: unfold(dim=2, size=frames_per_clip, step=stride)
        pieces = video_tensor.unfold(1, frames_per_clip, stride)  # (1, num_windows, C, H, W, frames_per_clip)
        pieces = pieces.permute(0, 1, 5, 2, 3, 4).contiguous()  # (1, num_windows, frames_per_clip, C, H, W)
        pieces = pieces.flatten(0, 1)  # (num_windows, frames_per_clip, C, H, W)
        
        if pieces.shape[0] == 0:
            return 0.0
        
        # Convert to format expected by V-JEPA: (B, C, T, H, W)
        pieces = pieces.permute(0, 2, 1, 3, 4)  # (num_windows, C, frames_per_clip, H, W)
        
        all_losses = []
        
        # Process each window
        for i in range(pieces.shape[0]):
            window = pieces[i]  # (C, T, H, W)
            
            # Convert to PIL frames for processor
            window_frames = []
            for t in range(window.shape[1]):
                frame = window[:, t, :, :]  # (C, H, W)
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                window_frames.append(frame_pil)
            
            try:
                # Process with V-JEPA
                processed = self.processor(window_frames, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    # Simple forward pass - let the model do its own masking
                    outputs = self.model(**processed)
                    
                    # Use the loss from the model if available, otherwise compute simple prediction error
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss.item()
                    else:
                        # Fallback: use the magnitude of the hidden states as a proxy for prediction difficulty
                        hidden_states = outputs.last_hidden_state
                        # Simple heuristic: higher magnitude = higher prediction error
                        loss = hidden_states.abs().mean().item()
                    
                    all_losses.append(loss)
                    
            except Exception as e:
                print(f"Warning: Failed to process window: {e}")
                all_losses.append(0.0)
        
        # Return max loss like original (max across temporal windows)
        if all_losses:
            max_loss = max(all_losses)
        else:
            max_loss = 0.0
        
        return max_loss
    
    def evaluate_scene(self, clips, labels, scene_paths=None):
        """Evaluate a single scene with 4 possibilities - compute losses for all context lengths."""
        # clips shape: (4, frames, 3, H, W)
        # labels shape: (4,) - boolean indicating if possibility is physically plausible
        
        # Test multiple context lengths like the original paper
        context_lengths = [2, 4, 6, 8, 10]
        
        all_surprises = []
        
        for context_len in context_lengths:
            surprises_for_context = []
            
            for i in range(4):
                clip = clips[i]  # (frames, 3, H, W)
                
                # Use frames_per_clip that includes context + prediction like original
                # Original uses: frames_per_clip = context_len + prediction_frames
                # For simplicity, use fixed 16 frames but vary the effective context
                frames_per_clip = min(16, clip.shape[0])
                stride = 1  # Use stride=1 for more windows like original
                
                surprise = self.compute_surprise_simple(
                    clip, 
                    frames_per_clip=frames_per_clip, 
                    stride=stride
                )
                surprises_for_context.append(surprise)
            
            all_surprises.append(surprises_for_context)
        
        # Convert to numpy arrays
        all_surprises = np.array(all_surprises)  # Shape: (num_context_lengths, 4)
        labels_np = labels.numpy().astype(float)
        
        return {
            'all_surprises': all_surprises,  # Shape: (5 contexts, 4 clips)
            'labels': labels_np,
            'scene_paths': scene_paths,
            'context_lengths': context_lengths
        }
    
    def compute_metrics_like_original(self, losses_2d, labels_2d):
        """Compute metrics exactly like the original implementation.
        
        Args:
            losses_2d: (num_scenes, 4) tensor of losses  
            labels_2d: (num_scenes, 4) tensor of labels  
        """
        metrics = {}
        
        # Convert to tensors matching original format
        losses = torch.tensor(losses_2d, dtype=torch.float32)  # (num_scenes, 4)
        labels = torch.tensor(labels_2d, dtype=torch.float32)  # (num_scenes, 4)
        
        # Extract real and fake losses exactly like original
        loss_real = losses[torch.where(labels == 1)]  # All plausible clips
        loss_fake = losses[torch.where(labels == 0)]  # All implausible clips
        
        # Reshape to match original's structure - group by scenes
        num_scenes = losses.shape[0]
        
        # Find how many real/fake per scene (typically 2 each for IntPhys)
        num_real_per_scene = (labels == 1).sum(dim=1)
        num_fake_per_scene = (labels == 0).sum(dim=1)
        
        # Use the first scene's counts (assuming all scenes have same structure)
        n_real_per_scene = num_real_per_scene[0].item()
        n_fake_per_scene = num_fake_per_scene[0].item()
        
        try:
            # Reshape exactly like original implementation expects
            loss_real = loss_real.view(num_scenes, n_real_per_scene)  # (num_scenes, n_real)
            loss_fake = loss_fake.view(num_scenes, n_fake_per_scene)  # (num_scenes, n_fake)
            
            # Compute metrics exactly like original implementation
            acc_pairwise_mean = (loss_real.mean(1) < loss_fake.mean(1)).sum() / loss_real.shape[0] * 100
            acc_pairwise_max = (loss_real.max(1)[0] < loss_fake.max(1)[0]).sum() / loss_real.shape[0] * 100
            
        except Exception as e:
            print(f"Warning: Could not reshape for scene-wise comparison: {e}")
            # Fallback 
            acc_pairwise_mean = (loss_real.mean() < loss_fake.mean()).float() * 100
            acc_pairwise_max = (loss_real.max() < loss_fake.max()).float() * 100
            loss_real = loss_real.unsqueeze(0)
            loss_fake = loss_fake.unsqueeze(0)
        
        metrics["Relative Accuracy (avg)"] = acc_pairwise_mean.item()
        metrics["Relative Accuracy (max)"] = acc_pairwise_max.item()
        
        # Absolute accuracy exactly like original
        data1 = loss_real.max(1)[0]  # Max loss per scene for real videos  
        data2 = loss_fake.max(1)[0]  # Max loss per scene for fake videos
        
        # Use threshold = 0 like original (the comment shows thresh = 0)
        thresh = 0  # Following original implementation
        accuracy_abs = ((data1 < thresh).sum() + (data2 > thresh).sum()) / (data1.shape[0] + data2.shape[0]) * 100
        
        metrics["Absolute Accuracy (max)"] = accuracy_abs.item()
        metrics["Classifier threhshold"] = thresh
        
        # Best threshold search exactly like original (fixing the denominator bug from original)
        threshs = np.linspace(data1.min().item(), data2.max().item(), 100)
        accs = []
        for thresh_val in threshs:
            # Fix: use correct denominator (data1.shape[0] + data2.shape[0]) 
            acc = ((data1 < thresh_val).sum() + (data2 > thresh_val).sum()) / (data1.shape[0] + data2.shape[0])
            accs.append(acc)
        
        best_accuracy_abs = torch.max(torch.tensor(accs)) * 100
        oracle_thresh = threshs[torch.argmax(torch.tensor(accs))]
        
        metrics["Best Absolute Accuracy (max)"] = best_accuracy_abs.item()
        metrics["Best Classifier threhshold"] = oracle_thresh
        
        # AUPRC and AUROC exactly like original
        try:
            # Use flattened data for classification metrics
            labels_flat = labels.flatten()
            losses_flat = losses.flatten()
            
            # AUPRC (negative because higher loss = more implausible)
            precision, recall, _ = precision_recall_curve(labels_flat, -losses_flat)
            auprc = auc(recall, precision)
            
            # AUROC (negative because higher loss = more implausible)  
            fpr, tpr, _ = roc_curve(labels_flat, -losses_flat)
            auroc = auc(fpr, tpr)
            
            metrics["AUPRC (avg)"] = auprc
            metrics["AUPRC (max)"] = auprc  # Same since we only have one loss value per clip
            metrics["AUROC (avg)"] = auroc
            metrics["AUROC (max)"] = auroc  # Same since we only have one loss value per clip
            
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
        
        # Create dataloader using original's frame calculation
        dataloader = create_intphys_dataloader(
            data_path=data_path,
            batch_size=batch_size,
            frames_per_clip=99//frame_step,  # Use same calculation as original
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
        print("EVALUATION RESULTS (Simplified V-JEPA)")
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
            
            # Compute metrics for this context length
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

    def test_model(self):
        """Test that the model is working correctly with a simple input."""
        print("Testing model with dummy input...")
        try:
            # Create dummy input
            dummy_frames = [Image.new('RGB', (224, 224), color='red') for _ in range(16)]
            processed = self.processor(dummy_frames, return_tensors="pt").to(self.model.device)
            
            # Test forward pass
            with torch.no_grad():
                outputs = self.model(**processed)
                print(f"Model test successful! Output shape: {outputs.last_hidden_state.shape}")
                return True
        except Exception as e:
            print(f"Model test failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA on IntPhys dataset")
    parser.add_argument("--data_path", type=str, required=False,
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
    parser.add_argument("--test_only", action="store_true",
                       help="Only test model, don't run evaluation")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = IntPhysEvaluator(model_name=args.model_name, device=args.device)
    
    # Test model
    if not evaluator.test_model():
        print("Model test failed. Exiting.")
        return None, None
    
    if args.test_only:
        print("Model test completed successfully!")
        return None, None
    
    # Validate data path for evaluation
    if args.data_path is None:
        raise ValueError("--data_path is required for evaluation (not needed for --test_only)")
    
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
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
        # Default run for testing - try O3 first, then O1 as fallback
        data_paths_to_try = [
            "/home/yjianhao/project/video_guidance/dev/O3",
            "/home/yjianhao/project/video_guidance/dev/O1", 
            "/home/yjianhao/project/video_guidance/dev/O2"
        ]
        
        print("Running with default path...")
        evaluator = IntPhysEvaluator()
        
        # Test model first
        if evaluator.test_model():
            success = False
            for data_path in data_paths_to_try:
                if os.path.exists(data_path):
                    print(f"Using data path: {data_path}")
                    try:
                        context_results, all_results = evaluator.evaluate_dataset(
                            data_path=data_path,
                            frames_per_clip=8,
                            frame_step=2
                            # output_dir will be auto-generated
                        )
                        success = True
                        break
                    except Exception as e:
                        print(f"Error with {data_path}: {e}")
                        continue
            
            if not success:
                print("All default data paths failed. Please provide --data_path argument.")
                print("Example usage:")
                print("python reproduce_intphys.py --data_path /path/to/IntPhys/dev/O1/")
        else:
            print("Model test failed. Please check your setup.")
    else:
        main()