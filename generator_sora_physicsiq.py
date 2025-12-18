"""
Physics-IQ video generation using Sora API.

This script loads Physics-IQ prompts and generates videos using Sora.
Generates N samples per prompt for Best-of-N (BoN) selection with VJEPA.

Usage:
    # Generate 3 samples per prompt
    python generator_sora_physicsiq.py --num_samples 3

    # With sharding for parallel execution
    python generator_sora_physicsiq.py --num_samples 3 --num_workers 8 --worker_idx 0
"""

import os
import json
import argparse
import math
from pipelines.sora_client import SoraClient


# Physics-IQ data location
PHYSICS_IQ_BASE_DIR = "/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark/physics-IQ-benchmark"


def load_physics_iq_data(json_path: str) -> list[dict]:
    """Load Physics-IQ dataset entries."""
    with open(json_path, 'r') as f:
        entries = json.load(f)
    return entries


def chunk_entries(entries: list, num_chunks: int, chunk_idx: int) -> list:
    """Divide entries into chunks for parallel processing."""
    chunk_size = math.ceil(len(entries) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(entries))
    return entries[start_idx:end_idx]


def generate_experiment_name(args) -> str:
    """Generate experiment folder name matching CogVideoX naming convention."""
    # Format: sora2_t2v_s{seconds}_{size}_n{num_samples}
    name = f"sora2_t2v_s{args.seconds}_{args.size}_n{args.num_samples}"
    return name


def main():
    parser = argparse.ArgumentParser(description="Physics-IQ video generation with Sora")
    
    # Data paths
    parser.add_argument('--batch_json', type=str, default='./prompts/physics_iq.json',
                       help='Path to Physics-IQ JSON file')
    parser.add_argument('--base_dir', type=str, default=PHYSICS_IQ_BASE_DIR,
                       help='Base directory for Physics-IQ benchmark data')
    parser.add_argument('--output_folder', type=str, default='./generated_videos/physics_iq/sora',
                       help='Output folder for generated videos')
    
    # Sora parameters (matched to CogVideoX: 49 frames @ 8fps ≈ 6s, 720x480)
    parser.add_argument('--model', type=str, default='sora-2',
                       help='Model to use')
    parser.add_argument('--seconds', type=int, default=8, choices=[4, 8, 12],
                       help='Video duration in seconds (CogVideoX uses ~6s, so 8s is closest)')
    parser.add_argument('--size', type=str, default='1280x720',
                       help='Video size (WxH): 1280x720 or 720x1280')
    
    # Sampling parameters (matched to CogVideoX rejection_samples=10)
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to generate per prompt (for BoN)')
    
    # Sharding for parallel execution
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Total number of parallel workers')
    parser.add_argument('--worker_idx', type=int, default=0,
                       help='Index of this worker (0 to num_workers-1)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading Physics-IQ data from: {args.batch_json}")
    entries = load_physics_iq_data(args.batch_json)
    print(f"Total entries: {len(entries)}")
    
    # Shard entries for parallel processing
    if args.num_workers > 1:
        entries = chunk_entries(entries, args.num_workers, args.worker_idx)
        print(f"Worker {args.worker_idx}/{args.num_workers}: Processing {len(entries)} entries")
    
    # Initialize client
    client = SoraClient()
    
    # Create output folder
    experiment_name = generate_experiment_name(args)
    output_folder = os.path.join(args.output_folder, experiment_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Print config
    print(f"\n{'='*60}")
    print(f"PHYSICS-IQ SORA GENERATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Duration: {args.seconds}s")
    print(f"Size: {args.size}")
    print(f"Samples per prompt: {args.num_samples}")
    print(f"Output: {output_folder}")
    print(f"Entries to process: {len(entries)}")
    print(f"{'='*60}\n")
    
    # Track progress
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, entry in enumerate(entries):
        prompt = entry.get('prompt')
        output_video = entry.get('output_video')
        
        if not prompt or not output_video:
            print(f"[{i+1}/{len(entries)}] Skipping: missing prompt or output_video")
            skip_count += 1
            continue
        
        # Generate N samples per prompt
        for sample_idx in range(args.num_samples):
            # Build output filename
            base_name = os.path.splitext(output_video)[0]
            if args.num_samples > 1:
                video_filename = f"{base_name}_sample{sample_idx}.mp4"
            else:
                video_filename = output_video
            
            output_path = os.path.join(output_folder, video_filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                print(f"[{i+1}/{len(entries)}][{sample_idx+1}/{args.num_samples}] Exists: {video_filename}")
                skip_count += 1
                continue
            
            print(f"[{i+1}/{len(entries)}][{sample_idx+1}/{args.num_samples}] Generating: {prompt[:60]}...")
            
            try:
                client.generate_and_download(
                    prompt=prompt,
                    output_path=output_path,
                    model=args.model,
                    seconds=args.seconds,
                    size=args.size,
                    verbose=True,
                )
                success_count += 1
                
            except Exception as e:
                print(f"  [error] {e}")
                error_count += 1
                continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Output: {output_folder}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

