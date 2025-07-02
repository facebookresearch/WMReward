from pipelines.wan_pipeline import WanPipeline
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video
from compute_vjepa_score import get_score, get_sliding_window_score, get_sliding_window_score_max, calculate_torch_vjepa_loss # Import the process_video function
from utils import init_torch_vjepa, preprocess_video_for_torch_vjepa
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
generator = torch.Generator(device="cuda").manual_seed(0)

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
context_length = 2  # Set context length for scoring

# Parameters for rejection sampling
num_attempts = 10
print(f"Running rejection sampling with {num_attempts} attempts using torch V-JEPA")
print(f"Context length: {context_length}")
print(f"Model: Quentin's torch V-JEPA implementation")

best_score = float('inf')
best_frames = None
worst_score = float('-inf')
worst_frames = None
all_scores = []

for i in range(num_attempts):  # Try 10 times
    frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames, num_inference_steps=50, generator=generator).frames[0]
    
    # Use torch V-JEPA with corrected preprocessing
    video_tensor = preprocess_video_for_torch_vjepa(frames)
    score = calculate_torch_vjepa_loss(video_tensor, torch_model, context_length=context_length)
    all_scores.append(score)
    
    export_to_video(frames, f"{dir_path}/wan-t2v{i}_{score:.6f}.mp4", fps=16)  # Export to video
    print(f"Attempt {i+1}/{num_attempts}: Score = {score:.6f}")
    
    if score < best_score:  # Update the best score and frames
        best_score = score
        best_frames = frames

    if score > worst_score:  # Update the worst score and frames
        worst_score = score
        worst_frames = frames

# Plot score distribution
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_attempts + 1), all_scores, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=best_score, color='g', linestyle='--', label=f'Best: {best_score:.6f}')
plt.axhline(y=worst_score, color='r', linestyle='--', label=f'Worst: {worst_score:.6f}')
plt.axhline(y=np.mean(all_scores), color='orange', linestyle='--', label=f'Mean: {np.mean(all_scores):.6f}')
plt.title('V-JEPA Loss Scores Across Rejection Sampling Attempts (Torch Version)')
plt.xlabel('Attempt')
plt.ylabel('V-JEPA Loss Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{dir_path}/score_distribution_torch.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nResults Summary:")
print(f"Best Score: {best_score:.6f}")
print(f"Worst Score: {worst_score:.6f}")
print(f"Mean Score: {np.mean(all_scores):.6f}")
print(f"Score Range: {worst_score - best_score:.6f}")

export_to_video(best_frames, f"{dir_path}/best_{best_score:.6f}.mp4", fps=16)  # Export the best frames to video
export_to_video(worst_frames,f"{dir_path}/worst_{worst_score:.6f}.mp4", fps=16)  # Export the worst frames to video

print(f"\nFiles saved to: {dir_path}")
print(f"- Score distribution plot: score_distribution_torch.png")
print(f"- Best video: best_{best_score:.6f}.mp4")
print(f"- Worst video: worst_{worst_score:.6f}.mp4")