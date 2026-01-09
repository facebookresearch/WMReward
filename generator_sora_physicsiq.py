#!/usr/bin/env python3
"""
Physics-IQ video generation using Sora API.
Breadth-first: ensures all prompts have N samples before moving to N+1.

Usage:
    python generator_sora_physicsiq.py --mode i2v --num_samples 8
    python generator_sora_physicsiq.py --mode i2v --num_workers 2 --worker_idx 0
"""

import os
import json
import argparse
import math
from PIL import Image
from pipelines.sora_client import SoraClient

PHYSICS_IQ_BASE = "/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark"


def load_first_frame(image_path: str = None, video_path: str = None) -> Image.Image:
    """Load first frame from image file or video."""
    if image_path and os.path.exists(image_path):
        return Image.open(image_path)
    
    if video_path and os.path.exists(video_path):
        try:
            from decord import VideoReader
            vr = VideoReader(video_path)
            frame = vr[0].asnumpy()
            return Image.fromarray(frame)
        except ImportError:
            try:
                import imageio
                reader = imageio.get_reader(video_path)
                frame = reader.get_data(0)
                reader.close()
                return Image.fromarray(frame)
            except ImportError:
                pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_json', default='./prompts/physics_iq.json')
    parser.add_argument('--base_dir', default=PHYSICS_IQ_BASE)
    parser.add_argument('--output_folder', default='./generated_videos/physics_iq/sora')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--mode', default='i2v', choices=['t2v', 'i2v'])
    parser.add_argument('--seconds', type=int, default=8, choices=[4, 8, 12])
    parser.add_argument('--size', default='1280x720')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--worker_idx', type=int, default=0)
    args = parser.parse_args()
    
    # Load all entries
    with open(args.batch_json) as f:
        all_entries = json.load(f)
    
    # Output directory
    exp_name = f"sora2_{args.mode}_s{args.seconds}_{args.size}_n{args.num_samples}"
    output_dir = args.output_dir if args.output_dir else os.path.join(args.output_folder, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PHYSICS-IQ SORA {args.mode.upper()} GENERATION (BREADTH-FIRST)")
    print(f"{'='*60}")
    print(f"Total prompts: {len(all_entries)} | Max samples: {args.num_samples}")
    print(f"Worker: {args.worker_idx}/{args.num_workers} | Output: {output_dir}")
    print(f"{'='*60}\n")
    
    client = SoraClient()
    stats = {"success": 0, "skip": 0, "error": 0}
    
    # Preload input images for I2V mode (cache to avoid reloading)
    entry_images = {}
    if args.mode == "i2v":
        print("Preloading input images...")
        for i, entry in enumerate(all_entries):
            img_path = entry.get("input_image")
            vid_path = entry.get("input_video")
            if img_path:
                img_path = os.path.join(args.base_dir, img_path)
            if vid_path:
                vid_path = os.path.join(args.base_dir, vid_path)
            entry_images[i] = load_first_frame(img_path, vid_path)
        valid = sum(1 for img in entry_images.values() if img is not None)
        print(f"Loaded {valid}/{len(all_entries)} images\n")
    
    # BREADTH-FIRST: iterate by sample index first, then by prompt
    # This ensures all prompts get sample 0, then all get sample 1, etc.
    for sample_idx in range(args.num_samples):
        print(f"\n{'='*60}")
        print(f"SAMPLE {sample_idx} / {args.num_samples}")
        print(f"{'='*60}")
        
        for entry_idx, entry in enumerate(all_entries):
            # Shard across workers by entry index
            if entry_idx % args.num_workers != args.worker_idx:
                continue
            
            prompt = entry.get("prompt")
            output_video = entry.get("output_video")
            
            if not prompt or not output_video:
                continue
            
            # Build output path
            base_name = os.path.splitext(output_video)[0]
            out_name = f"{base_name}_sample{sample_idx}.mp4"
            out_path = os.path.join(output_dir, out_name)
            
            # Skip if exists
            if os.path.exists(out_path):
                stats["skip"] += 1
                continue
            
            # Get input image for I2V
            input_img = None
            if args.mode == "i2v":
                input_img = entry_images.get(entry_idx)
                if not input_img:
                    stats["skip"] += 1
                    continue
            
            label = "I2V" if args.mode == "i2v" else "T2V"
            print(f"[S{sample_idx}][{entry_idx+1}/{len(all_entries)}] [{label}] {prompt[:50]}...")
            
            try:
                client.generate_and_download(
                    prompt=prompt,
                    output_path=out_path,
                    input_reference=input_img,
                    seconds=args.seconds,
                    size=args.size,
                )
                stats["success"] += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                stats["error"] += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {stats['success']} success, {stats['skip']} skip, {stats['error']} error")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
