from pipelines.wan_pipeline import WanPipeline
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video
import torch
import matplotlib.pyplot as plt
import numpy as np
from compute_vjepa_score import get_score, get_sliding_window_score, get_sliding_window_score_max # Import the process_video function
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
generator = torch.Generator(device="cuda").manual_seed(0)

# prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
# prompt="a ball falling to the ground"
# tag='ball_drop'
# prompt = "A robot arm moving towards a red cube"
# tag = 'robot_arm_cube_vith_nt10_kernel16_context12_stride2_max_rej10_lossarr'

# prompt = "a truck accelerating to gain speed"
# tag="truck_accelerate_vith_nt10_kernel8_context4_stride2_max_rej10_lossarr"

# prompt = "A person running through a park on a sunny day, with trees and flowers in the background"
# tag = "person_running_park_nt10_kernel4_context2_stride2_max_rej10_lossarr"

prompt = "a cat playing in park"
tag = "cat_playing_park_nt10_kernel4_context2_stride2_max_rej10_lossarr"

dir_path = f"./visualization/{tag}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation"
num_frames = 33

# init vjepa
jepa_model_id = "facebook/vjepa2-vith-fpc64-256"  # Use the ViT-L model for better performance
processor = AutoVideoProcessor.from_pretrained(jepa_model_id)
model = AutoModel.from_pretrained(
    jepa_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

best_score = float('inf')
best_frames = None
worst_score = float('-inf')
worst_frames = None
loss_arr = None
for i in range(10):  # Try 10 times
    frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames, num_inference_steps=50, generator=generator).frames[0]
    
    # score = get_score(frames, model, processor, n_timesteps=10)  # Calculate the score
    # score = get_sliding_window_score(frames, model, processor, context_length=12, kernel_size=16, stride=4)  
    score, loss_arr = get_sliding_window_score_max(frames, model, processor, kernel_size=4, context_window_size=2, stride=2, loss_form="max", return_loss_arr=True)
    export_to_video(frames, f"{dir_path}/wan-t2v{i}_{score}.mp4", fps=16)  # Export to video
    print(f"{i}, Score: {score}")
    print(f"Loss Array: {loss_arr}")
    
    if score < best_score:  # Update the best score and frames
        best_score = score
        best_frames = frames

    if score > worst_score:  # Update the worst score and frames
        worst_score = score
        worst_frames = frames

    if loss_arr is not None:
        plt.figure(figsize=(10,6))
        plt.plot(loss_arr)
        # Set title and labels
        plt.title('Loss Over Time')
        plt.xlabel('timestep')
        plt.ylabel('Loss')
        # Save the plot locally
        plt.savefig(f"{dir_path}/loss_{i}_{score}.png")


print(f"Best Score: {best_score}")
export_to_video(best_frames, f"{dir_path}/best_{best_score}.mp4", fps=16)  # Export the best frames to video

print(f"Worst Score: {worst_score}")
export_to_video(worst_frames,f"{dir_path}/worst_{worst_score}.mp4", fps=16)  # Export the worst frames to video