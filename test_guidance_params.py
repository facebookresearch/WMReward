from pipelines.wan_pipeline_guidance_v2 import WanPipeline
from schedulers.unipc_multistep_scheduler import UniPCMultistepScheduler
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video
import torch
import os
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def test_guidance_parameters(quick_test=False):
    """Test different rho_scale values and guidance ranges for a single prompt."""
    
    # Configuration
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    
    # Single test prompt
    prompt = "A robot arm reaching for a red cube, the robot arm is moving towards the red cube, the robot arm is in a factory setting, the robot arm is metallic and shiny, the red cube is on a conveyor belt, the robot arm is precise and controlled, the scene is dynamic and industrial"
    negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation"
    
    # Video generation parameters
    num_frames = 33
    height, width = 480, 832
    num_inference_steps = 50
    cfg_scale = 5.0
    
    # V-JEPA parameters for guidance
    kernel_size = 8
    context_length = 6
    stride = 2
    vjepa_mode = "mean"
    
    # Parameter ranges to test
    if quick_test:
        # Quick test - fewer combinations for rapid experimentation
        rho_scale_values = [1.0, 3.0, 5.0]
        guidance_ranges = [
            (0, 1001),     # Full range
            (0, 500),      # First half
            (500, 1001),   # Second half
        ]
        output_suffix = "_quick"
    else:
        # Comprehensive test - full parameter sweep
        rho_scale_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        guidance_ranges = [
            (0, 1001),     # Full range (default)
            (0, 500),      # First half
            (250, 750),    # Middle range
            (500, 1001),   # Second half
            (0, 250),      # First quarter
            (750, 1001),   # Last quarter
        ]
        output_suffix = "_full"
    
    # Create organized output directory structure
    output_dir = Path(f"./guidance_param_test{output_suffix}")
    videos_dir = output_dir / "videos"
    data_dir = output_dir / "loss_data"
    plots_dir = output_dir / "plots"
    
    for dir_path in [output_dir, videos_dir, data_dir, plots_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.scheduler = scheduler
    pipe.enable_model_cpu_offload()
    
    # Set base V-JEPA parameters
    pipe.vjepa_kernel_size = kernel_size
    pipe.vjepa_context_length = context_length
    pipe.vjepa_stride = stride
    pipe.vjepa_mode = vjepa_mode
    
    # Generate videos for all parameter combinations
    total_combinations = len(rho_scale_values) * len(guidance_ranges)
    current_combination = 0
    all_loss_data = {}  # Store all loss trajectories for visualization
    
    test_mode = "QUICK" if quick_test else "FULL"
    print(f"\n{test_mode} TEST: {total_combinations} parameter combinations")
    print(f"Prompt: {prompt}")
    print(f"Rho scales: {rho_scale_values}")
    print(f"Guidance ranges: {guidance_ranges}")
    print(f"Output structure:")
    print(f"  ├─ videos/     - Generated MP4 files")
    print(f"  ├─ loss_data/  - JSON files with loss trajectories")
    print(f"  └─ plots/      - Loss visualization plots")
    print("-" * 80)
    
    # Test all combinations
    for rho_scale in rho_scale_values:
        for guidance_start, guidance_end in guidance_ranges:
            current_combination += 1
            
            # Set guidance parameters
            pipe.guidance_rho_scale = rho_scale
            pipe.guidance_start = guidance_start
            pipe.guidance_end = guidance_end
            
            # Generate filenames
            config_name = f"rho{rho_scale}_range{guidance_start}-{guidance_end}"
            video_filename = f"{config_name}.mp4"
            loss_filename = f"{config_name}_losses.json"
            
            video_filepath = videos_dir / video_filename
            loss_filepath = data_dir / loss_filename
            
            # Skip if video already exists
            if video_filepath.exists():
                print(f"[{current_combination:2d}/{total_combinations}] Skipping existing: {config_name}")
                # Load existing loss data for visualization if it exists
                if loss_filepath.exists():
                    with open(loss_filepath, 'r') as f:
                        all_loss_data[config_name] = json.load(f)
                continue
            
            print(f"[{current_combination:2d}/{total_combinations}] Generating: {config_name}")
            print(f"  ├─ Rho scale: {rho_scale}")
            print(f"  ├─ Guidance range: {guidance_start}-{guidance_end}")
            
            # Generate video and track losses
            generator = torch.Generator(device="cuda").manual_seed(42)  # Fixed seed for consistency
            
            # Enable loss tracking in the pipeline
            pipe.track_losses = True
            pipe.loss_history = []
            
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg_scale
            )
            
            frames = result.frames[0]
            
            # Get loss trajectory from pipeline
            loss_trajectory = getattr(pipe, 'loss_history', [])
            timesteps = list(range(len(loss_trajectory)))
            
            # Filter out None values for guidance-only losses
            guidance_losses = [l for l in loss_trajectory if l is not None]
            guidance_timesteps = [i for i, l in enumerate(loss_trajectory) if l is not None]
            
            # Save loss data
            loss_data = {
                'config': {
                    'rho_scale': rho_scale,
                    'guidance_start': guidance_start,
                    'guidance_end': guidance_end,
                    'num_inference_steps': num_inference_steps
                },
                'timesteps': timesteps,
                'losses': loss_trajectory,  # Full trajectory with None values
                'guidance_only_losses': guidance_losses,  # Only guidance losses
                'guidance_timesteps': guidance_timesteps,  # Timesteps where guidance was applied
                'final_loss': guidance_losses[-1] if guidance_losses else None
            }
            
            with open(loss_filepath, 'w') as f:
                json.dump(loss_data, f, indent=2)
            
            # Store for visualization
            all_loss_data[config_name] = loss_data
            
            # Export video
            export_to_video(frames, str(video_filepath), fps=16)
            
            print(f"  ├─ Video saved: {video_filepath.name}")
            print(f"  ├─ Loss data saved: {loss_filepath.name}")
            if loss_trajectory and loss_trajectory[-1] is not None:
                print(f"  └─ Final loss: {loss_trajectory[-1]:.4f}")
            else:
                print(f"  └─ No valid loss data captured")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED!")
    print(f"Results saved in: {output_dir.absolute()}")
    
    # List generated content
    video_files = sorted(videos_dir.glob("*.mp4"))
    loss_files = sorted(data_dir.glob("*.json"))
    
    print(f"\nGenerated {len(video_files)} videos:")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    print(f"\nSaved {len(loss_files)} loss trajectories")
    
    # Create visualizations
    if all_loss_data:
        print("\nCreating loss visualizations...")
        create_loss_visualizations(all_loss_data, plots_dir, rho_scale_values, guidance_ranges)
    
    # Create summary report
    create_summary_report(output_dir, videos_dir, data_dir, plots_dir, rho_scale_values, guidance_ranges, all_loss_data)

def create_loss_visualizations(all_loss_data, plots_dir, rho_scale_values, guidance_ranges):
    """Create visualizations of loss trajectories."""
    
    # Plot 1: All trajectories together
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_loss_data)))
    
    for i, (config_name, data) in enumerate(all_loss_data.items()):
        if data.get('guidance_only_losses'):
            timesteps = data['guidance_timesteps']
            losses = data['guidance_only_losses']
            config_info = data['config']
            label = f"ρ={config_info['rho_scale']}, range={config_info['guidance_start']}-{config_info['guidance_end']}"
            plt.plot(timesteps, losses, color=colors[i], label=label, linewidth=2, alpha=0.8)
    
    plt.xlabel('Timestep')
    plt.ylabel('V-JEPA Loss')
    plt.title('V-JEPA Loss Trajectories - All Configurations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'all_loss_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: By rho_scale
    plt.figure(figsize=(15, 5))
    unique_rho_scales = sorted(set(data['config']['rho_scale'] for data in all_loss_data.values()))
    
    for i, rho_scale in enumerate(unique_rho_scales):
        plt.subplot(1, len(unique_rho_scales), i + 1)
        rho_data = {k: v for k, v in all_loss_data.items() if v['config']['rho_scale'] == rho_scale}
        
        for config_name, data in rho_data.items():
            if data.get('guidance_only_losses'):
                config_info = data['config']
                label = f"range={config_info['guidance_start']}-{config_info['guidance_end']}"
                plt.plot(data['guidance_timesteps'], data['guidance_only_losses'], label=label, linewidth=2)
        
        plt.xlabel('Timestep')
        plt.ylabel('V-JEPA Loss')
        plt.title(f'Rho Scale = {rho_scale}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_by_rho_scale.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: By guidance range
    plt.figure(figsize=(20, 10))
    unique_ranges = sorted(set((data['config']['guidance_start'], data['config']['guidance_end']) 
                              for data in all_loss_data.values()))
    
    n_cols = 3
    n_rows = (len(unique_ranges) + n_cols - 1) // n_cols
    
    for i, (start, end) in enumerate(unique_ranges):
        plt.subplot(n_rows, n_cols, i + 1)
        range_data = {k: v for k, v in all_loss_data.items() 
                     if v['config']['guidance_start'] == start and v['config']['guidance_end'] == end}
        
        for config_name, data in range_data.items():
            if data.get('guidance_only_losses'):
                config_info = data['config']
                label = f"ρ={config_info['rho_scale']}"
                plt.plot(data['guidance_timesteps'], data['guidance_only_losses'], label=label, linewidth=2)
        
        plt.xlabel('Timestep')
        plt.ylabel('V-JEPA Loss')
        plt.title(f'Guidance Range: {start}-{end}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_by_guidance_range.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Final loss summary
    plt.figure(figsize=(12, 8))
    final_losses = []
    config_labels = []
    
    for config_name, data in all_loss_data.items():
        if data.get('final_loss') is not None:
            final_losses.append(data['final_loss'])
            config_info = data['config']
            label = f"ρ={config_info['rho_scale']}\nrange={config_info['guidance_start']}-{config_info['guidance_end']}"
            config_labels.append(label)
    
    x_pos = np.arange(len(final_losses))
    bars = plt.bar(x_pos, final_losses, alpha=0.7)
    
    # Color bars by rho_scale
    rho_scales = [all_loss_data[list(all_loss_data.keys())[i]]['config']['rho_scale'] for i in range(len(final_losses))]
    unique_rhos = sorted(set(rho_scales))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_rhos)))
    rho_color_map = {rho: colors[i] for i, rho in enumerate(unique_rhos)}
    
    for bar, rho in zip(bars, rho_scales):
        bar.set_color(rho_color_map[rho])
    
    plt.xlabel('Configuration')
    plt.ylabel('Final V-JEPA Loss')
    plt.title('Final Loss by Configuration')
    plt.xticks(x_pos, config_labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'final_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ├─ Saved: all_loss_trajectories.png")
    print(f"  ├─ Saved: loss_by_rho_scale.png")
    print(f"  ├─ Saved: loss_by_guidance_range.png")
    print(f"  └─ Saved: final_loss_comparison.png")

def create_summary_report(output_dir, videos_dir, data_dir, plots_dir, rho_scale_values, guidance_ranges, all_loss_data):
    """Create a comprehensive summary report of the parameter combinations tested."""
    
    report_path = output_dir / "parameter_test_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("GUIDANCE PARAMETER TEST SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Test Configuration:\n")
        f.write(f"  - Rho scale values: {rho_scale_values}\n")
        f.write(f"  - Guidance ranges: {guidance_ranges}\n")
        f.write(f"  - Total combinations: {len(rho_scale_values) * len(guidance_ranges)}\n\n")
        
        f.write("Directory Structure:\n")
        f.write(f"  ├─ videos/     - {len(list(videos_dir.glob('*.mp4')))} MP4 files\n")
        f.write(f"  ├─ loss_data/  - {len(list(data_dir.glob('*.json')))} JSON files with loss trajectories\n")
        f.write(f"  └─ plots/      - {len(list(plots_dir.glob('*.png')))} visualization plots\n\n")
        
        f.write("Parameter Combinations and Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Configuration':<30} {'Video':<8} {'Loss Data':<10} {'Final Loss':<12}\n")
        f.write("-" * 50 + "\n")
        
        for rho_scale in rho_scale_values:
            for guidance_start, guidance_end in guidance_ranges:
                config_name = f"rho{rho_scale}_range{guidance_start}-{guidance_end}"
                video_file = videos_dir / f"{config_name}.mp4"
                loss_file = data_dir / f"{config_name}_losses.json"
                
                video_status = "✓" if video_file.exists() else "✗"
                loss_status = "✓" if loss_file.exists() else "✗"
                
                final_loss = "N/A"
                if config_name in all_loss_data and all_loss_data[config_name].get('final_loss'):
                    final_loss = f"{all_loss_data[config_name]['final_loss']:.4f}"
                
                f.write(f"{config_name:<30} {video_status:<8} {loss_status:<10} {final_loss:<12}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Loss Analysis Summary:\n")
        if all_loss_data:
            # Find best and worst configurations by final loss
            configs_with_loss = [(k, v['final_loss']) for k, v in all_loss_data.items() 
                               if v.get('final_loss') is not None]
            
            if configs_with_loss:
                best_config = min(configs_with_loss, key=lambda x: x[1])
                worst_config = max(configs_with_loss, key=lambda x: x[1])
                avg_loss = np.mean([loss for _, loss in configs_with_loss])
                
                f.write(f"  - Best configuration (lowest final loss): {best_config[0]} ({best_config[1]:.4f})\n")
                f.write(f"  - Worst configuration (highest final loss): {worst_config[0]} ({worst_config[1]:.4f})\n")
                f.write(f"  - Average final loss: {avg_loss:.4f}\n")
                f.write(f"  - Loss range: {best_config[1]:.4f} - {worst_config[1]:.4f}\n\n")
                
                # Analysis by rho_scale
                f.write("Loss by Rho Scale:\n")
                for rho in sorted(set(rho_scale_values)):
                    rho_losses = [loss for config, loss in configs_with_loss 
                                if all_loss_data[config]['config']['rho_scale'] == rho]
                    if rho_losses:
                        f.write(f"  - ρ={rho}: avg={np.mean(rho_losses):.4f}, "
                               f"min={min(rho_losses):.4f}, max={max(rho_losses):.4f}\n")
        else:
            f.write("  - No loss data available\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Analysis Guidelines:\n")
        f.write("1. Video Quality Assessment:\n")
        f.write("   - Compare videos with different rho_scale values\n")
        f.write("   - Lower rho_scale = gentler guidance, higher = stronger guidance\n")
        f.write("   - Look for over-smoothing or artifacts with high rho values\n\n")
        
        f.write("2. Temporal Guidance Effects:\n")
        f.write("   - Early guidance (0-500): affects initial structure/composition\n")
        f.write("   - Late guidance (500-1001): refines details and motion\n")
        f.write("   - Full range (0-1001): comprehensive guidance throughout\n\n")
        
        f.write("3. Loss Trajectory Analysis:\n")
        f.write("   - Check plots/ folder for visualizations\n")
        f.write("   - Look for steady loss decrease vs. fluctuations\n")
        f.write("   - Compare final losses but also trajectory shapes\n")
        f.write("   - Lower final loss doesn't always mean better visual quality\n\n")
        
        f.write("4. Recommended Next Steps:\n")
        f.write("   - Review loss_by_rho_scale.png for optimal rho values\n")
        f.write("   - Check loss_by_guidance_range.png for timing effects\n")
        f.write("   - Watch videos corresponding to best/worst loss trajectories\n")
        f.write("   - Consider human evaluation alongside loss metrics\n")
    
    print(f"\nComprehensive summary report saved: {report_path}")

if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Running QUICK test (9 combinations)...")
        test_guidance_parameters(quick_test=True)
    else:
        print("Running FULL test (30 combinations)...")
        print("Use '--quick' flag for faster testing with fewer combinations")
        test_guidance_parameters(quick_test=False) 