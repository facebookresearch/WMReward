#!/usr/bin/env python3
"""
Best-of-N Analysis with Bootstrap Sampling

For N in [1, 2, 4, 8, 16]:
- N=16: Use all samples
- N<16: Bootstrap 5 times with random sample selection
- Run best-of-N, trim to 5s, evaluate Physics-IQ
- Plot results with error bars
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
from collections import defaultdict
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/home/reyhaneaskari/WMReward')

import torch
from compute_wmreward import load_vjepa_models, load_video_as_tensor
from utils import compute_vjepa_loss_sliding_window


def get_all_prompts(input_dir):
    """Get list of unique prompts from video filenames."""
    prompts = set()
    for f in os.listdir(input_dir):
        if f.endswith('.mp4'):
            # Format: XXXX_name_sampleN.mp4
            parts = f.rsplit('_sample', 1)
            if len(parts) == 2:
                prompts.add(parts[0])
    return sorted(prompts)


def get_samples_for_prompt(input_dir, prompt, max_samples=16):
    """Get all sample paths for a given prompt."""
    samples = []
    for s in range(max_samples):
        path = os.path.join(input_dir, f"{prompt}_sample{s}.mp4")
        if os.path.exists(path):
            samples.append(path)
    return samples


def select_best_video(video_paths, encoder, target_encoder, predictor, device, img_size=256):
    """Select the best video from a list using V-JEPA scoring."""
    best_path = None
    best_score = float('inf')
    
    for path in video_paths:
        try:
            video = load_video_as_tensor(path, max_frames=49, img_size=img_size)
            if video is None:
                continue
            video = video.to(device)
            
            with torch.no_grad():
                loss = compute_vjepa_loss_sliding_window(
                    video_tensor=video, encoder=encoder, target_encoder=target_encoder, predictor=predictor,
                    img_size=img_size, window_size=16, context_frames=8, stride=2
                )
            
            if loss < best_score:
                best_score = loss
                best_path = path
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    return best_path, best_score


def trim_video_to_5s(input_path, output_path):
    """Trim video to exactly 5 seconds using ffmpeg (no audio to ensure exact duration)."""
    ffmpeg_path = '/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/ffmpeg'
    cmd = [
        ffmpeg_path, '-y', '-i', input_path,
        '-vf', 'fps=30', '-vframes', '150',
        '-an',  # No audio - ensures exactly 5.000s duration
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_physics_iq(input_folder, output_folder):
    """Run Physics-IQ evaluation and return the score."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Use conda environment Python
    python_path = '/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/python'
    
    cmd = [
        python_path, 'code/run_physics_iq.py',
        '--input_folders', input_folder,
        '--output_folder', output_folder,
        '--descriptions_file', '/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark/descriptions/descriptions.csv'
    ]
    
    # Set PATH to include conda env binaries (including ffprobe)
    env = os.environ.copy()
    conda_bin = '/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin'
    env['PATH'] = f'{conda_bin}:{env.get("PATH", "")}'
    
    # Run from Physics-IQ directory
    result = subprocess.run(
        cmd,
        cwd='/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark',
        capture_output=True,
        text=True,
        env=env
    )
    
    # Debug: print any errors
    if result.returncode != 0:
        print(f"  Physics-IQ returned code {result.returncode}")
        print(f"  stderr: {result.stderr[:500] if result.stderr else 'None'}")
    
    # Parse score from output
    for line in result.stdout.split('\n'):
        if 'Physics-IQ score' in line:
            try:
                score = float(line.split(':')[-1].strip())
                return score
            except:
                pass
    
    return None


def run_bootstrap_analysis(args):
    """Run the full bootstrap analysis."""
    
    print("=" * 60)
    print("BEST-OF-N BOOTSTRAP ANALYSIS")
    print("=" * 60)
    
    # Setup directories
    base_dir = Path(args.base_dir)
    input_8s = base_dir / "sora2_i2v_s8_1280x720_n16"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load V-JEPA model
    print("\nLoading V-JEPA model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, target_encoder, predictor, img_size = load_vjepa_models(args.model)
    encoder = encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    
    # Get all prompts
    prompts = get_all_prompts(str(input_8s))
    print(f"Found {len(prompts)} prompts")
    
    # N values to test
    if args.n_value:
        n_values = [args.n_value]
    else:
        n_values = [1, 2, 4, 8, 16]
    num_bootstrap = args.num_bootstrap
    
    # Results storage
    results = {n: [] for n in n_values}
    
    for n in n_values:
        print(f"\n{'='*60}")
        print(f"N = {n}")
        print("=" * 60)
        
        if n == 16:
            # No bootstrap needed, use all samples
            iterations = 1
        else:
            iterations = num_bootstrap
        
        for bootstrap_idx in range(iterations):
            print(f"\n--- Bootstrap {bootstrap_idx + 1}/{iterations} ---")
            
            # Create output directories for this run
            run_name = f"n{n}_b{bootstrap_idx}"
            best_dir = results_dir / f"best_{run_name}"
            trimmed_dir = results_dir / f"trimmed_{run_name}"
            eval_dir = results_dir / f"eval_{run_name}"
            
            best_dir.mkdir(exist_ok=True)
            trimmed_dir.mkdir(exist_ok=True)
            
            # Process each prompt
            for i, prompt in enumerate(prompts):
                if (i + 1) % 20 == 0:
                    print(f"  Processing prompt {i + 1}/{len(prompts)}")
                
                # Get all samples for this prompt
                all_samples = get_samples_for_prompt(str(input_8s), prompt, max_samples=16)
                
                if len(all_samples) < n:
                    print(f"  Warning: {prompt} has only {len(all_samples)} samples, skipping")
                    continue
                
                # Select N samples (randomly for bootstrap, or all for N=16)
                if n == 16:
                    selected_samples = all_samples
                else:
                    random.seed(bootstrap_idx * 1000 + i)  # Reproducible randomness
                    selected_samples = random.sample(all_samples, n)
                
                # Find best video
                if n == 1:
                    # For N=1, just use the selected sample
                    best_path = selected_samples[0]
                else:
                    best_path, _ = select_best_video(selected_samples, encoder, target_encoder, predictor, device, img_size)
                
                if best_path is None:
                    continue
                
                # Copy best to output
                output_name = os.path.basename(best_path).rsplit('_sample', 1)[0] + '.mp4'
                shutil.copy(best_path, best_dir / output_name)
                
                # Trim to 5s
                trim_video_to_5s(str(best_dir / output_name), str(trimmed_dir / output_name))
            
            # Run Physics-IQ evaluation
            print(f"  Running Physics-IQ evaluation...")
            # Use absolute paths for Physics-IQ
            score = run_physics_iq(str(trimmed_dir.resolve()), str(eval_dir.resolve()))
            
            if score is not None:
                results[n].append(score)
                print(f"  Score: {score:.2f}")
            else:
                print(f"  Warning: Could not get Physics-IQ score")
    
    # Save results (per N value to avoid overwrites)
    for n in n_values:
        results_file = results_dir / f"bootstrap_results_n{n}.json"
        with open(results_file, 'w') as f:
            json.dump({n: results[n]}, f, indent=2)
        print(f"Results for N={n} saved to {results_file}")
    
    # Create summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = {}
    for n in n_values:
        scores = results[n]
        if scores:
            mean = np.mean(scores)
            std = np.std(scores) if len(scores) > 1 else 0
            summary[n] = {'mean': mean, 'std': std, 'scores': scores}
            print(f"N={n:2d}: {mean:.2f} ± {std:.2f} (n={len(scores)} runs)")
    
    # Create plot
    create_plot(summary, results_dir / "best_of_n_analysis.pdf")
    create_plot(summary, results_dir / "best_of_n_analysis.png")
    
    return summary


def create_plot(summary, output_path):
    """Create a publication-quality plot of results."""
    
    plt.figure(figsize=(8, 6))
    
    n_values = sorted(summary.keys())
    means = [summary[n]['mean'] for n in n_values]
    stds = [summary[n]['std'] for n in n_values]
    
    # Plot with error bars
    plt.errorbar(n_values, means, yerr=stds, 
                 fmt='o-', capsize=5, capthick=2, 
                 markersize=10, linewidth=2,
                 color='#2E86AB', ecolor='#A23B72')
    
    # Add data labels
    for n, mean, std in zip(n_values, means, stds):
        if std > 0:
            label = f'{mean:.1f}±{std:.1f}'
        else:
            label = f'{mean:.1f}'
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
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./generated_videos/physics_iq/sora',
                        help='Base directory with video folders')
    parser.add_argument('--results_dir', default='./results/physics_iq/bootstrap_analysis',
                        help='Output directory for results')
    parser.add_argument('--model', default='vitg', choices=['vith', 'vitg', 'vitg384'])
    parser.add_argument('--num_bootstrap', type=int, default=5,
                        help='Number of bootstrap iterations for N<16')
    parser.add_argument('--n_value', type=int, default=None,
                        help='Specific N value to test (if None, tests all)')
    args = parser.parse_args()
    
    run_bootstrap_analysis(args)

