from pipelines.wan_pipeline_guidance import WanPipeline
from schedulers.unipc_multistep_scheduler import UniPCMultistepScheduler
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
scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.scheduler = scheduler
pipe.enable_model_cpu_offload()
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()
# pipe.vae.enable_gradient_checkpointing()

# print(pipe.scheduler)
prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 17

frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]

export_to_video(worst_frames,f"./temp/guidance_sample.mp4", fps=16)  # Export the worst frames to video