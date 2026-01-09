#!/usr/bin/env python3
"""
Fast Best-of-N Analysis using pre-computed V-JEPA scores.
Reuses scores from bon_results.json instead of re-computing.
"""

import os
import sys
import json
import random
import shutil
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_bon_results(json_path):
    """Load pre-computed V-JEPA scores."""
    with open(json_path) as f:
        return json.load(f)


def select_best_from_n(prompt_data, n, bootstrap_seed=None):
    """Select best video from N samples using pre-computed scores."""
    all_scores = prompt_data['all_scores']  # List of [filename, score]
    
    if n >= len(all_scores):
        # Use all samples
        selected = all_scores
    else:
        if bootstrap_seed is not None:
            # Random selection for bootstrap
            random.seed(bootstrap_seed)
            selected = random.sample(all_scores, n)
        else:
            # Take first N samples (sorted by filename)
            sorted_scores = sorted(all_scores, key=lambda x: x[0])
            selected = sorted_scores[:n]
    
    # Find best (lowest loss)
    best = min(selected, key=lambda x: x[1])
    return best[0], best[1]


def trim_video_to_5s(input_path, output_path):
    """Trim video to exactly 5 seconds (no audio for exact duration)."""
    ffmpeg = '/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/ffmpeg'
    cmd = [ffmpeg, '-y', '-i', input_path, '-vf', 'fps=30', '-vframes', '150',
           '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_physics_iq(input_folder, output_folder):
    """Run Physics-IQ evaluation."""
    os.makedirs(output_folder, exist_ok=True)
    
    python_path = '/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/python'
    cmd = [
        python_path, 'code/run_physics_iq.py',
        '--input_folders', input_folder,
        '--output_folder', output_folder,
        '--descriptions_file', '/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark/descriptions/descriptions.csv'
    ]
    
    env = os.environ.copy()
    env['PATH'] = '/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin:' + env.get('PATH', '')
    
    print(f"    Calling: {' '.join(cmd[:3])}...", flush=True)
    
    result = subprocess.run(
        cmd,
        cwd='/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark',
        capture_output=True, text=True, env=env
    )
    
    # Parse score
    for line in result.stdout.split('\n'):
        if 'Physics-IQ score' in line:
            try:
                return float(line.split(':')[-1].strip())
            except:
                pass
    
    # Debug output on failure
    print(f"    Physics-IQ returncode: {result.returncode}", flush=True)
    if result.stderr:
        print(f"    stderr: {result.stderr[:500]}", flush=True)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bon_json', default='./generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16_8s_best/bon_results.json')
    parser.add_argument('--source_dir', default='./generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16')
    parser.add_argument('--results_dir', default='./results/physics_iq/bon_analysis')
    parser.add_argument('--num_bootstrap', type=int, default=5)
    parser.add_argument('--n_value', type=int, default=None, help='Specific N to run')
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("BEST-OF-N ANALYSIS FROM PRE-COMPUTED SCORES", flush=True)
    print("=" * 60, flush=True)
    
    # Load pre-computed scores
    print(f"\nLoading scores from: {args.bon_json}", flush=True)
    bon_data = load_bon_results(args.bon_json)
    print(f"Loaded {len(bon_data)} prompts with scores", flush=True)
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # N values to test
    if args.n_value:
        n_values = [args.n_value]
    else:
        n_values = [1, 2, 4, 8, 16]
    
    results = {}
    
    for n in n_values:
        print(f"\n{'='*60}", flush=True)
        print(f"N = {n}", flush=True)
        print("=" * 60, flush=True)
        
        iterations = 1 if n == 16 else args.num_bootstrap
        scores = []
        
        for b in range(iterations):
            print(f"\n--- Bootstrap {b+1}/{iterations} ---", flush=True)
            
            # Output directories
            run_name = f"n{n}_b{b}"
            trimmed_dir = results_dir / f"trimmed_{run_name}"
            eval_dir = results_dir / f"eval_{run_name}"
            trimmed_dir.mkdir(exist_ok=True)
            
            # Select best videos and trim
            trim_count = 0
            skip_count = 0
            for prompt_data in bon_data:
                prompt = prompt_data['prompt']
                seed = b * 1000 + hash(prompt) % 1000 if n < 16 else None
                best_file, best_loss = select_best_from_n(prompt_data, n, seed)
                
                # Source and output paths
                src = Path(args.source_dir) / best_file
                # Output name: remove _sampleN suffix but keep full original prefix
                # best_file is like "0001_perspective-left_trimmed-ball-and-block-fall_sample7.mp4"
                # We want "0001_perspective-left_trimmed-ball-and-block-fall.mp4"
                out_name = best_file.rsplit('_sample', 1)[0] + '.mp4'
                dst = trimmed_dir / out_name
                
                if not dst.exists():
                    trim_video_to_5s(str(src), str(dst))
                    trim_count += 1
                else:
                    skip_count += 1
            
            trimmed_count = len(list(trimmed_dir.glob('*.mp4')))
            print(f"  Trimmed: {trim_count} new, {skip_count} skipped, {trimmed_count} total", flush=True)
            
            # Run Physics-IQ
            print("  Running Physics-IQ...", flush=True)
            sys.stdout.flush()
            
            score = run_physics_iq(str(trimmed_dir.resolve()), str(eval_dir.resolve()))
            
            if score:
                scores.append(score)
                print(f"  Score: {score:.2f}", flush=True)
            else:
                print("  WARNING: No score returned", flush=True)
        
        results[n] = scores
        
        # Save results
        with open(results_dir / f"results_n{n}.json", 'w') as f:
            json.dump({n: scores}, f, indent=2)
        print(f"  Saved results to results_n{n}.json", flush=True)
    
    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    summary = {}
    for n in n_values:
        if results.get(n):
            mean = np.mean(results[n])
            std = np.std(results[n]) if len(results[n]) > 1 else 0
            summary[n] = {'mean': float(mean), 'std': float(std), 'scores': results[n]}
            print(f"N={n:2d}: {mean:.2f} ± {std:.2f}", flush=True)
    
    # Save full summary
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create plot
    if len(summary) > 1:
        create_plot(summary, results_dir / "best_of_n_analysis.pdf")
        create_plot(summary, results_dir / "best_of_n_analysis.png")


def create_plot(summary, output_path):
    """Create plot of results."""
    plt.figure(figsize=(8, 6))
    
    n_values = sorted(summary.keys())
    means = [summary[n]['mean'] for n in n_values]
    stds = [summary[n]['std'] for n in n_values]
    
    plt.errorbar(n_values, means, yerr=stds, 
                 fmt='o-', capsize=5, capthick=2, 
                 markersize=10, linewidth=2,
                 color='#2E86AB', ecolor='#A23B72')
    
    for n, mean, std in zip(n_values, means, stds):
        label = f'{mean:.1f}±{std:.1f}' if std > 0 else f'{mean:.1f}'
        plt.annotate(label, (n, mean), textcoords="offset points", 
                    xytext=(0, 12), ha='center', fontsize=10)
    
    plt.xlabel('N (Number of Samples)', fontsize=14)
    plt.ylabel('Physics-IQ Score', fontsize=14)
    plt.title('Best-of-N Selection: Physics-IQ Performance', fontsize=16)
    plt.xscale('log', base=2)
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}", flush=True)
    plt.close()


if __name__ == '__main__':
    main()
