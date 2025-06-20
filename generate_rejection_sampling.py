from pipelines.wan_pipeline import WanPipeline
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video
import torch
from compute_vjepa_score import get_score  # Import the process_video function
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
prompt="a ball falls to the ground"
tag='ball_drop'
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

# init vjepa
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

best_score = float('inf')
best_frames = None
worst_score = float('-inf')
worst_frames = None

for i in range(10):  # Try 10 times
    frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
    
    score = get_score(frames, model, processor)  # Calculate the score
    export_to_video(frames, f"./temp/{tag}/wan-t2v{i}_{score}.mp4", fps=16)  # Export to video
    print(f"{i}, Score: {score}")
    
    if score < best_score:  # Update the best score and frames
        best_score = score
        best_frames = frames

    if score > worst_score:  # Update the worst score and frames
        worst_score = score
        worst_frames = frames

print(f"Best Score: {best_score}")
export_to_video(best_frames, f"./temp/{tag}/bes_{score}.mp4", fps=16)  # Export the best frames to video

print(f"Worst Score: {worst_score}")
export_to_video(worst_frames,f"./temp/{tag}/worst_{score}.mp4", fps=16)  # Export the worst frames to video