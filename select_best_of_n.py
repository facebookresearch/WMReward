#!/usr/bin/env python3
"""
Select best-of-N videos using V-JEPA scoring.
Reuses existing compute_wmreward.py functions.

Usage:
    python select_best_of_n.py --input_dir ./generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16
"""

import os
import glob
import json
import argparse
import shutil
from collections import defaultdict

import torch
from compute_wmreward import load_vjepa_models, load_video_as_tensor
from utils import compute_vjepa_loss_sliding_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with *_sampleN.mp4 videos')
    parser.add_argument('--output_dir', default=None, help='Output dir for best videos (default: input_dir + _best)')
    parser.add_argument('--model', default='vitg', choices=['vith', 'vitg', 'vitg384'])
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--context_frames', type=int, default=8)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--dry_run', action='store_true', help='Only compute scores, do not copy')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or args.input_dir + "_best"
    os.makedirs(output_dir, exist_ok=True)

    # Load V-JEPA models once
    print(f"Loading V-JEPA model: {args.model}...")
    encoder, target_encoder, predictor, img_size = load_vjepa_models(args.model)
    encoder, target_encoder, predictor = encoder.to(device).eval(), target_encoder.to(device).eval(), predictor.to(device).eval()

    # Group videos by prompt (base name without _sampleN)
    videos = sorted(glob.glob(os.path.join(args.input_dir, "*_sample*.mp4")))
    groups = defaultdict(list)
    for v in videos:
        base = os.path.basename(v).rsplit('_sample', 1)[0]
        groups[base].append(v)

    print(f"Found {len(videos)} videos across {len(groups)} prompts\n")

    results = []
    for i, (base, samples) in enumerate(sorted(groups.items())):
        print(f"[{i+1}/{len(groups)}] {base} ({len(samples)} samples)")
        
        best_path, best_loss = None, float('inf')
        sample_scores = []

        for vid_path in sorted(samples):
            try:
                video_tensor = load_video_as_tensor(vid_path, max_frames=49, img_size=img_size).to(device)
                with torch.no_grad():
                    loss = compute_vjepa_loss_sliding_window(
                        video_tensor=video_tensor,
                        encoder=encoder, target_encoder=target_encoder, predictor=predictor,
                        img_size=img_size, window_size=args.window_size,
                        masking_mode="causal", context_frames=args.context_frames,
                        is_vae_output=True, stride=args.stride, mode="mean"
                    ).item()
                sample_scores.append((os.path.basename(vid_path), loss))
                if loss < best_loss:
                    best_loss, best_path = loss, vid_path
                print(f"  {os.path.basename(vid_path)}: {loss:.6f}")
            except Exception as e:
                print(f"  {os.path.basename(vid_path)}: ERROR - {e}")

        if best_path and not args.dry_run:
            out_name = base + ".mp4"
            shutil.copy(best_path, os.path.join(output_dir, out_name))
            print(f"  -> Best: {os.path.basename(best_path)} (loss={best_loss:.6f})")

        results.append({"prompt": base, "best": os.path.basename(best_path) if best_path else None, 
                       "best_loss": best_loss, "all_scores": sample_scores})

    # Save results
    results_path = os.path.join(output_dir, "bon_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Best videos copied to: {output_dir}")


if __name__ == "__main__":
    main()
