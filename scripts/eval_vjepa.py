#!/usr/bin/env python3
"""
V-JEPA Surprise Loss Evaluation Script

This script calculates V-JEPA surprise scores for videos.
"""

import os
import argparse
import csv
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path to import utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_vjepa_models_torchhub,
    load_vjepa_model_source,
    compute_vjepa_loss_sliding_window,
    get_video,
    set_deterministic
)

class VJEPASurpriseEvaluator:
    """V-JEPA surprise loss evaluator."""
    
    def __init__(self, model_name: str = "vitg", use_source: bool = False, device: str = "cuda"):
        """
        Initialize V-JEPA surprise evaluator.
        
        Args:
            model_name: V-JEPA model to use ('vith', 'vitg', 'vitg384', 'vitgac')
            use_source: Whether to use source models or torch.hub models
            device: Device to run models on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading V-JEPA model: {model_name}")
        # if use_source:
        #     self.encoder, self.target_encoder, self.predictor, self.img_size = load_vjepa_model_source(model_name)
        # else:
        self.encoder, self.target_encoder, self.predictor, self.img_size = load_vjepa_models_torchhub(model_name)
        
        # Move models to device
        self.encoder = self.encoder.to(self.device)
        self.target_encoder = self.target_encoder.to(self.device)
        self.predictor = self.predictor.to(self.device)
        
        # Set to eval mode
        self.encoder.eval()
        self.target_encoder.eval()
        self.predictor.eval()
        
        print(f"Model loaded successfully. Image size: {self.img_size}")
    
    def calculate_surprise_score(
        self,
        video_path: str,
        window_size: int = 16,
        stride: int = 2,
        context_frames: int = 8,
        masking_mode: str = "causal",
        mode: str = "mean",
    ) -> dict:
        """
        Calculate V-JEPA surprise score for a single video.
        
        Args:
            video_path: Path to video file
            **kwargs: Additional arguments for surprise calculation
            
        Returns:
            Dictionary containing surprise score and metadata
        """
        # Load video
        video = get_video(video_path, max_frames=49)
        # video shape: [T, H, W, C] -> [B, C, T, H, W]
        video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)
        video_tensor = video_tensor.to(self.device)
        
        
        print(f"Video tensor shape: {video_tensor.shape}")
        print(f"Video tensor min: {video_tensor.min()}, max: {video_tensor.max()}")

        # Calculate surprise score
        with torch.no_grad():
            surprise_score = compute_vjepa_loss_sliding_window(
                video_tensor=video_tensor,
                encoder=self.encoder,
                target_encoder=self.target_encoder,
                predictor=self.predictor,
                img_size=self.img_size,
                window_size=window_size,
                masking_mode=masking_mode,
                context_frames=context_frames,
                is_vae_output=False,  # Our video is in [0, 255] range, not VAE output
                stride=stride,
                mode=mode,
            )
        
        return {
            'success': True,
            'surprise_score': surprise_score.item(),
            'similarity_score': 1 - surprise_score.item(),
            'video_path': video_path,
            'video_shape': list(video_tensor.shape),
            'model_name': self.model_name,
            'img_size': self.img_size
        }
    
    def batch_surprise_evaluation(
        self,
        folder_path: str,
        output_file: str = "vjepa_evaluation.csv",
        window_size: int = 16,
        stride: int = 4,
        context_frames: int = 8,
        masking_mode: str = "causal",
        mode: str = "max",
    ) -> dict:
        """
        Evaluate surprise scores for all videos in a folder.
        
        Args:
            folder_path: Path to folder containing videos
            output_file: Output CSV file path
            **kwargs: Additional arguments for surprise calculation
            
        Returns:
            Dictionary containing evaluation summary
        """
        folder = Path(folder_path)
        video_files = list(folder.glob("*.mp4")) + list(folder.glob("*.avi")) + list(folder.glob("*.mov"))
        
        if not video_files:
            return {'error': f"No video files found in {folder_path}"}
        
        print(f"Found {len(video_files)} video files")
        
        results = []
        successful = 0
        failed = 0
        
        for video_file in video_files:
            print(f"Processing: {video_file.name}")
            result = self.calculate_surprise_score(
                str(video_file),
                window_size=window_size,
                stride=stride,
                context_frames=context_frames,
                masking_mode=masking_mode,
                mode=mode,
            )
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Save results to CSV
        self._save_results_to_csv(results, output_file)
        
        # Generate summary
        summary = self._generate_evaluation_summary(results)
        
        return summary
    
    def _save_results_to_csv(self, results: list, output_file: str):
        """Save evaluation results to CSV file."""
        if not results:
            return
        
        # Determine all possible fields
        all_fields = set()
        for result in results:
            all_fields.update(result.keys())
        
        # Order fields logically
        field_order = [
            'success', 'video_path', 'surprise_score', 'similarity_score',
            'model_name', 'img_size', 'video_shape', 'error'
        ]
        
        # Add any remaining fields
        for field in sorted(all_fields):
            if field not in field_order:
                field_order.append(field)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {output_file}")
    
    def _generate_evaluation_summary(self, results: list) -> dict:
        """Generate summary statistics from evaluation results."""
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        summary = {
            'total_videos': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) if results else 0
        }
        
        if successful_results:
            surprise_scores = [r['surprise_score'] for r in successful_results]
            similarity_scores = [r['similarity_score'] for r in successful_results]
            
            summary.update({
                'surprise_score': {
                    'mean': np.mean(surprise_scores),
                    'std': np.std(surprise_scores),
                    'min': np.min(surprise_scores),
                    'max': np.max(surprise_scores),
                    'median': np.median(surprise_scores)
                },
                'similarity_score': {
                    'mean': np.mean(similarity_scores),
                    'std': np.std(similarity_scores),
                    'min': np.min(similarity_scores),
                    'max': np.max(similarity_scores),
                    'median': np.median(similarity_scores)
                }
            })
        
        if failed_results:
            summary['errors'] = [r.get('error', 'Unknown error') for r in failed_results]
        
        return summary

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="V-JEPA Surprise Loss Evaluation")
    parser.add_argument("--video-path", help="Path to video file (for single video)")
    parser.add_argument("--folder-path", help="Path to folder (for batch processing)")
    parser.add_argument("--output", default="vjepa_evaluation.csv", help="Output file path")
    parser.add_argument("--model", default="vitg", choices=["vith", "vitg", "vitg384", "vitgac"],
                       help="V-JEPA model to use")
    parser.add_argument("--use-source", action="store_true",
                       help="Use source models instead of torch.hub models")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    
    # Surprise evaluation arguments
    parser.add_argument("--window-size", type=int, default=16, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=4, help="Sliding window stride")
    parser.add_argument("--context-frames", type=int, default=8, help="Context frames")
    parser.add_argument("--masking-mode", default="causal", choices=["causal", "block", "random"])
    parser.add_argument("--mode", default="max", choices=["mean", "max"])
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = VJEPASurpriseEvaluator(
        model_name=args.model,
        use_source=args.use_source,
        device=args.device
    )
    
    # Check if we're doing batch or single video
    if args.folder_path:
        # Batch processing
        print(f"Processing folder: {args.folder_path}")
        summary = evaluator.batch_surprise_evaluation(
            args.folder_path,
            args.output,
            window_size=args.window_size,
            stride=args.stride,
            context_frames=args.context_frames,
            masking_mode=args.masking_mode,
            mode=args.mode
        )
        
        if 'error' not in summary:
            print("\nEvaluation Summary:")
            print(f"Total videos: {summary['total_videos']}")
            print(f"Successful: {summary['successful']}")
            print(f"Failed: {summary['failed']}")
            print(f"Success rate: {summary['success_rate']:.2%}")
            if 'surprise_score' in summary:
                print(f"Surprise score - Mean: {summary['surprise_score']['mean']:.6f}, Std: {summary['surprise_score']['std']:.6f}")
        else:
            print(f"Error: {summary['error']}")
    
    elif args.video_path:
        # Single video processing
        print(f"Processing video: {args.video_path}")
        result = evaluator.calculate_surprise_score(
            args.video_path,
            window_size=args.window_size,
            stride=args.stride,
            context_frames=args.context_frames,
            masking_mode=args.masking_mode,
            mode=args.mode
        )
        
        if result['success']:
            print(f"Surprise Score: {result['surprise_score']:.6f}")
            print(f"Similarity Score: {result['similarity_score']:.6f}")
        else:
            print(f"Error: {result['error']}")
    
    else:
        print("Error: Must specify either --video-path or --folder-path")
        return

if __name__ == "__main__":
    main()
