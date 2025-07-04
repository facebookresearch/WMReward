from pipelines.wan_pipeline import WanPipeline
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video
from compute_vjepa_score import calculate_torch_vjepa_loss # Import the process_video function
from utils import init_torch_vjepa, preprocess_video_for_torch_vjepa
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def create_video_strip(frames, num_frames_to_show=8):
    """Create a horizontal strip of video frames"""
    total_frames = len(frames)
    if num_frames_to_show >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames-1, num_frames_to_show, dtype=int)
    
    # Convert frames to numpy arrays and concatenate horizontally
    selected_frames = []
    for i in frame_indices:
        frame = frames[i]
        
        # Handle different frame formats
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
        elif isinstance(frame, np.ndarray):
            frame_np = frame.copy()
            
            # Handle data type conversion
            if frame_np.dtype == np.float32 or frame_np.dtype == np.float64:
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)
            elif frame_np.dtype != np.uint8:
                frame_np = frame_np.astype(np.uint8)
            
            # Ensure frame has proper dimensions
            if len(frame_np.shape) == 4:  # (1, H, W, C) - squeeze batch dimension
                frame_np = frame_np.squeeze(0)
            elif len(frame_np.shape) == 2:  # (H, W) - add channel dimension
                frame_np = np.stack([frame_np] * 3, axis=-1)
        else:
            frame_np = np.array(frame)
        
        # Ensure we have a valid frame shape (H, W, C)
        if len(frame_np.shape) == 3 and frame_np.shape[-1] in [1, 3, 4]:
            selected_frames.append(frame_np)
        else:
            print(f"Warning: Skipping frame {i} with unexpected shape {frame_np.shape}")
            # Create a placeholder frame if needed
            if selected_frames:
                placeholder = np.zeros_like(selected_frames[0])
                selected_frames.append(placeholder)
    
    if selected_frames:
        video_strip = np.concatenate(selected_frames, axis=1)  # Concatenate along width
        return video_strip, frame_indices
    else:
        # Return empty frame if no valid frames
        empty_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        return empty_frame, []

def create_comprehensive_rejection_plot(candidates_data, prompt, save_path):
    """
    Create a comprehensive plot showing all rejection sampling candidates
    candidates_data: list of tuples (attempt_num, frames, loss_arr, score)
    """
    num_candidates = len(candidates_data)
    
    # Create figure with video strips on top and loss curves below
    fig = plt.figure(figsize=(20, 4 + 3 * num_candidates))
    
    # Create grid: video strips take 2/3 height, loss plots take 1/3
    gs = fig.add_gridspec(num_candidates + 1, 1, height_ratios=[2] * num_candidates + [3])
    
    # Colors for different candidates
    colors = plt.cm.tab10(np.linspace(0, 1, num_candidates))
    
    # Plot video strips
    for i, (attempt_num, frames, loss_arr, score) in enumerate(candidates_data):
        ax_video = fig.add_subplot(gs[i, 0])
        
        # Create video strip
        video_strip, frame_indices = create_video_strip(frames, num_frames_to_show=8)
        
        ax_video.imshow(video_strip)
        ax_video.set_title(f'Candidate {attempt_num} (Score: {score:.6f})', fontsize=12, fontweight='bold')
        ax_video.set_xticks([])
        ax_video.set_yticks([])
        
        # Add frame numbers
        frame_width = video_strip.shape[1] // len(frame_indices)
        for j, frame_idx in enumerate(frame_indices):
            x_pos = (j + 0.5) * frame_width
            ax_video.text(x_pos, video_strip.shape[0] + 10, f'F{frame_idx}', 
                         ha='center', va='bottom', fontsize=8)
    
    # Plot all loss curves together
    ax_loss = fig.add_subplot(gs[-1, 0])
    
    best_score = min([score for _, _, _, score in candidates_data])
    worst_score = max([score for _, _, _, score in candidates_data])
    
    for i, (attempt_num, frames, loss_arr, score) in enumerate(candidates_data):
        # Convert loss_arr to numpy if needed
        if torch.is_tensor(loss_arr):
            loss_arr_np = loss_arr.cpu().numpy().flatten()
        else:
            loss_arr_np = np.array(loss_arr)
        
        # Color code by performance
        if score == best_score:
            color = 'green'
            linewidth = 3
            alpha = 1.0
            label = f'Candidate {attempt_num} (BEST: {score:.6f})'
        elif score == worst_score:
            color = 'red'
            linewidth = 3
            alpha = 1.0
            label = f'Candidate {attempt_num} (WORST: {score:.6f})'
        else:
            color = colors[i]
            linewidth = 2
            alpha = 0.7
            label = f'Candidate {attempt_num} ({score:.6f})'
        
        ax_loss.plot(loss_arr_np, color=color, linewidth=linewidth, alpha=alpha,
                    label=label, marker='o', markersize=2)
    
    ax_loss.set_title(f'V-JEPA Loss Comparison - Rejection Sampling\nPrompt: "{prompt}"', 
                     fontsize=14, fontweight='bold')
    ax_loss.set_xlabel('Timestep', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add statistics
    scores = [score for _, _, _, score in candidates_data]
    stats_text = f'Best: {best_score:.6f}\nWorst: {worst_score:.6f}\nMean: {np.mean(scores):.6f}\nStd: {np.std(scores):.6f}'
    ax_loss.text(0.02, 0.98, stats_text, 
                transform=ax_loss.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive rejection sampling plot saved: {save_path}")

def create_summary_gif(candidates_data, prompt, save_path, duration=1500):
    """
    Create a comprehensive animated GIF that cycles through all candidates
    showing video frames and loss curves for each
    """
    temp_frames = []
    
    for i, (attempt_num, frames, loss_arr, score) in enumerate(candidates_data):
        # Create figure for this candidate
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1]})
        
        # Plot video frames on top
        video_strip, frame_indices = create_video_strip(frames, num_frames_to_show=8)
        ax1.imshow(video_strip)
        ax1.set_title(f'Candidate {attempt_num} Video Frames (Score: {score:.6f})', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Frame Sequence', fontsize=12)
        ax1.set_ylabel('Height', fontsize=12)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Add frame numbers as labels
        if len(frame_indices) > 0:
            frame_width = video_strip.shape[1] // len(frame_indices)
            for j, frame_idx in enumerate(frame_indices):
                x_pos = (j + 0.5) * frame_width
                ax1.text(x_pos, video_strip.shape[0] + 10, f'F{frame_idx}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Convert loss_arr to numpy if it's a tensor
        if torch.is_tensor(loss_arr):
            loss_arr_np = loss_arr.cpu().numpy().flatten()
        else:
            loss_arr_np = np.array(loss_arr)
        
        # Plot loss curve on bottom
        ax2.plot(loss_arr_np, 'b-', linewidth=3, marker='o', markersize=4)
        ax2.set_title(f'V-JEPA Loss Over Time (Candidate {attempt_num})', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add comprehensive annotation
        best_score = min([s for _, _, _, s in candidates_data])
        worst_score = max([s for _, _, _, s in candidates_data])
        rank = sorted([s for _, _, _, s in candidates_data]).index(score) + 1
        
        status = ""
        if score == best_score:
            status = " 🏆 BEST"
            ax2.set_facecolor('#e8f5e8')  # Light green background
        elif score == worst_score:
            status = " 🔻 WORST"
            ax2.set_facecolor('#ffe8e8')  # Light red background
        
        info_text = f'Candidate: {attempt_num}{status}\nScore: {score:.6f}\nRank: {rank}/{len(candidates_data)}\nPrompt: "{prompt}"'
        ax2.text(0.02, 0.98, info_text, 
                 transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # Add progress indicator
        progress_text = f"Showing {i+1}/{len(candidates_data)} candidates"
        fig.suptitle(f'Rejection Sampling Analysis - {progress_text}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        
        # Convert plot to image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        temp_frames.append(buf)
        
        plt.close(fig)
    
    # Create animated GIF
    if temp_frames:
        pil_frames = [Image.fromarray(frame) for frame in temp_frames]
        
        # Add a pause frame at the end showing summary
        summary_fig, ax = plt.subplots(figsize=(16, 10))
        scores = [score for _, _, _, score in candidates_data]
        attempts = [attempt_num for attempt_num, _, _, _ in candidates_data]
        
        # Create summary plot
        colors = ['green' if s == min(scores) else 'red' if s == max(scores) else 'blue' for s in scores]
        bars = ax.bar(attempts, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Rejection Sampling Summary\nPrompt: "{prompt}"', 
                    fontsize=18, fontweight='bold')
        ax.set_xlabel('Candidate', fontsize=14)
        ax.set_ylabel('V-JEPA Score', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Best: {min(scores):.6f}\nWorst: {max(scores):.6f}\nMean: {np.mean(scores):.6f}\nStd: {np.std(scores):.6f}\nRange: {max(scores) - min(scores):.6f}'
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        plt.tight_layout()
        
        # Convert summary to image
        summary_fig.canvas.draw()
        buf = np.frombuffer(summary_fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(summary_fig.canvas.get_width_height()[::-1] + (3,))
        summary_frame = Image.fromarray(buf)
        plt.close(summary_fig)
        
        # Add summary frame multiple times for longer pause
        pil_frames.extend([summary_frame] * 3)  # Show summary for 3x duration
        
        # Save animated GIF
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        
        print(f"Comprehensive summary GIF saved: {save_path}")
    else:
        print("Warning: No frames to create summary GIF")

def save_frames_as_gif(frames, gif_path, duration=200):
    """Save frames as animated GIF"""
    pil_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            pil_frames.append(frame)
        else:
            # Handle numpy arrays - convert to uint8 if needed
            if isinstance(frame, np.ndarray):
                # Convert float to uint8 if necessary
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    # Assume values are in [0, 1] range, scale to [0, 255]
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                elif frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Ensure proper shape for PIL
                if frame.shape[-1] == 3:  # RGB
                    pil_frames.append(Image.fromarray(frame))
                elif len(frame.shape) == 2:  # Grayscale
                    pil_frames.append(Image.fromarray(frame, mode='L'))
                else:
                    print(f"Warning: Unexpected frame shape {frame.shape}, skipping frame")
                    continue
            else:
                pil_frames.append(Image.fromarray(np.array(frame)))
    
    if pil_frames:
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
    else:
        print(f"Warning: No valid frames to save as GIF: {gif_path}")

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
# Don't fix the seed - we want different generations for rejection sampling
# generator = torch.Generator(device="cuda").manual_seed(0)

prompt = "a cat playing in park"
tag = "cat_playing_park_nt10_kernel4_context2_stride2_max_rej10_lossarr_torch"

dir_path = f"./visualization/{tag}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation"
num_frames = 33

# init torch vjepa (using corrected version)
print("Loading torch V-JEPA model...")
torch_model = init_torch_vjepa()
context_length = 10  # Set context length for scoring (changed from 2 to 4, which is more standard)

# Parameters for rejection sampling
num_attempts = 10
print(f"Running rejection sampling with {num_attempts} attempts using torch V-JEPA")
print(f"Context length: {context_length}")
print(f"Model: Quentin's torch V-JEPA implementation")

best_score = float('inf')
best_frames = None
best_loss_arr = None
worst_score = float('-inf')
worst_frames = None
worst_loss_arr = None
all_scores = []
all_candidates = []  # Store all candidate data for comprehensive visualization

for i in range(num_attempts):  # Try 10 times
    # Create a new random generator for each attempt to ensure different results
    generator = torch.Generator(device="cuda").manual_seed(i * 42 + 12345)  # Different seed each time
    frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames, num_inference_steps=50, generator=generator).frames[0]
    
    # Use torch V-JEPA with corrected preprocessing - get both score and loss array
    video_tensor = preprocess_video_for_torch_vjepa(frames)
    
    # Debug: Check if video tensors are actually different
    video_hash = hash(video_tensor.cpu().numpy().tobytes())
    video_mean = video_tensor.mean().item()
    video_std = video_tensor.std().item()
    print(f"  Video tensor hash: {video_hash}")
    print(f"  Video tensor stats: mean={video_mean:.6f}, std={video_std:.6f}")
    print(f"  Video tensor shape: {video_tensor.shape}")
    
    # Clear any potential CUDA cache before scoring
    torch.cuda.empty_cache()
    
    # Ensure model is in eval mode and reset any potential state
    torch_model.eval()
    
    # Check if model has any cached state that needs clearing
    if hasattr(torch_model, 'reset_state'):
        torch_model.reset_state()
    
    # Add some randomness to break any deterministic behavior
    torch.manual_seed(i * 1337 + int(video_mean * 10000))
    
    print(f"  Model training mode: {torch_model.training}")
    print(f"  Random seed set to: {i * 1337 + int(video_mean * 10000)}")
    
    # Try with slightly different parameters to break any potential caching
    score, loss_arr = calculate_torch_vjepa_loss(
                                video_tensor, 
                                torch_model,
                                context_length=context_length,
                                frames_per_clip=16,
                                stride=2,
                                use_bfloat16=True,
                                require_grad=False,
                                mode='max',
                                return_arr=True,  # Get the full loss array
                                is_vae_output=False  # Explicitly set this
                            )
    
    print(f"  Raw score before any processing: {score}")
    
    # Debug: Check loss array details
    if torch.is_tensor(loss_arr):
        loss_arr_np = loss_arr.cpu().numpy().flatten()
    else:
        loss_arr_np = np.array(loss_arr)
    
    print(f"  Loss array shape: {loss_arr_np.shape}")
    print(f"  Loss array stats: min={loss_arr_np.min():.6f}, max={loss_arr_np.max():.6f}, mean={loss_arr_np.mean():.6f}")
    print(f"  Loss array hash: {hash(loss_arr_np.tobytes())}")
    
    all_scores.append(score)
    all_candidates.append((i+1, frames, loss_arr, score))  # Store candidate data
    
    # Save individual files
    export_to_video(frames, f"{dir_path}/wan-t2v{i+1}_{score:.6f}.mp4", fps=16)
    save_frames_as_gif(frames, f"{dir_path}/wan-t2v{i+1}_{score:.6f}.gif", duration=150)
    
    # Create individual plot for this candidate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot video frames on top
    video_strip, frame_indices = create_video_strip(frames, num_frames_to_show=8)
    ax1.imshow(video_strip)
    ax1.set_title(f'Candidate {i+1} Video Frames (Score: {score:.6f})')
    ax1.set_xlabel('Frame Sequence')
    ax1.set_ylabel('Height')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add frame numbers as labels
    frame_width = video_strip.shape[1] // len(frame_indices)
    for j, frame_idx in enumerate(frame_indices):
        x_pos = (j + 0.5) * frame_width
        ax1.text(x_pos, video_strip.shape[0] + 10, f'F{frame_idx}', 
            ha='center', va='bottom', fontsize=8)
    
    # Convert loss_arr to numpy if it's a tensor
    if torch.is_tensor(loss_arr):
        loss_arr_np = loss_arr.cpu().numpy().flatten()
    else:
        loss_arr_np = loss_arr
    
    # Plot loss curve on bottom
    ax2.plot(loss_arr_np, 'b-', linewidth=2, marker='o', markersize=3)
    ax2.set_title(f'V-JEPA Loss Over Time (Candidate {i+1})')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    info_text = f'Candidate: {i+1}\nScore: {score:.6f}\nPrompt: "{prompt}"'
    ax2.text(0.02, 0.98, info_text, 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{dir_path}/candidate_{i+1}_{score:.6f}_analysis.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attempt {i+1}/{num_attempts}: Score = {score:.6f}")
    
    if score < best_score:  # Update the best score and frames
        best_score = score
        best_frames = frames
        best_loss_arr = loss_arr

    if score > worst_score:  # Update the worst score and frames
        worst_score = score
        worst_frames = frames
        worst_loss_arr = loss_arr

# Create comprehensive visualization showing all candidates
create_comprehensive_rejection_plot(all_candidates, prompt, f"{dir_path}/comprehensive_rejection_analysis.png")

# Create animated summary GIF cycling through all candidates
create_summary_gif(all_candidates, prompt, f"{dir_path}/SUMMARY_all_candidates.gif", duration=2000)

# Plot score distribution
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(range(1, num_attempts + 1), all_scores, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=best_score, color='g', linestyle='--', label=f'Best: {best_score:.6f}')
plt.axhline(y=worst_score, color='r', linestyle='--', label=f'Worst: {worst_score:.6f}')
plt.axhline(y=np.mean(all_scores), color='orange', linestyle='--', label=f'Mean: {np.mean(all_scores):.6f}')
plt.title(f'V-JEPA Loss Scores Across Rejection Sampling Attempts\nPrompt: "{prompt}"')
plt.xlabel('Attempt')
plt.ylabel('V-JEPA Loss Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Add histogram of scores
plt.subplot(2, 1, 2)
plt.hist(all_scores, bins=min(num_attempts//2, 10), alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=best_score, color='g', linestyle='--', label=f'Best: {best_score:.6f}')
plt.axvline(x=worst_score, color='r', linestyle='--', label=f'Worst: {worst_score:.6f}')
plt.axvline(x=np.mean(all_scores), color='orange', linestyle='--', label=f'Mean: {np.mean(all_scores):.6f}')
plt.title('Distribution of V-JEPA Scores')
plt.xlabel('V-JEPA Loss Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{dir_path}/score_distribution_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nResults Summary:")
print(f"Best Score: {best_score:.6f}")
print(f"Worst Score: {worst_score:.6f}")
print(f"Mean Score: {np.mean(all_scores):.6f}")
print(f"Std Score: {np.std(all_scores):.6f}")
print(f"Score Range: {worst_score - best_score:.6f}")

# Export best and worst videos with additional formats
export_to_video(best_frames, f"{dir_path}/BEST_{best_score:.6f}.mp4", fps=16)
export_to_video(worst_frames, f"{dir_path}/WORST_{worst_score:.6f}.mp4", fps=16)
save_frames_as_gif(best_frames, f"{dir_path}/BEST_{best_score:.6f}.gif", duration=150)
save_frames_as_gif(worst_frames, f"{dir_path}/WORST_{worst_score:.6f}.gif", duration=150)

print(f"\nFiles saved to: {dir_path}")
print(f"🎬 SUMMARY GIF:")
print(f"  - SUMMARY_all_candidates.gif (animated slideshow of all candidates)")
print(f"📊 Comprehensive Analysis:")
print(f"  - comprehensive_rejection_analysis.png (all candidates with loss curves)")
print(f"  - score_distribution_analysis.png (score trends and histogram)")
print(f"🎬 Individual Candidates:")
for i in range(num_attempts):
    score = all_scores[i]
    print(f"  - candidate_{i+1}_{score:.6f}_analysis.png")
    print(f"  - wan-t2v{i+1}_{score:.6f}.mp4/.gif")
print(f"🏆 Best/Worst:")
print(f"  - BEST_{best_score:.6f}.mp4/.gif")
print(f"  - WORST_{worst_score:.6f}.mp4/.gif")