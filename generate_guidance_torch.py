from pipelines.wan_pipeline_guidance_torch import WanPipeline
# from pipelines.wan_pipeline import WanPipeline
from schedulers.unipc_multistep_scheduler import UniPCMultistepScheduler
from diffusers.utils import export_to_video
import torch
from utils import init_torch_vjepa, preprocess_video_for_torch_vjepa  # Import torch V-JEPA functions
from compute_vjepa_score import calculate_torch_vjepa_loss  # Import torch V-JEPA loss computation
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Initialize torch V-JEPA model
print("🚀 Loading torch V-JEPA model...")
torch_vjepa = init_torch_vjepa()

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe.scheduler = scheduler

# Replace HuggingFace V-JEPA with torch version in the pipeline
pipe.vjepa_model = torch_vjepa
pipe.vjepa_processor = None

# Set V-JEPA parameters to match config (frames_per_clip = 16)
pipe.vjepa_kernel_size = 16

pipe.enable_model_cpu_offload()
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()
# pipe.vae.enable_gradient_checkpointing()

generator = torch.Generator(device="cuda").manual_seed(42)
# prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
# prompt="a truck accelerating to gain speed"
# prompt = "A robot arm moving towards a red cube"
prompt = "A robot arm reaching for a red cube, the robot arm is moving towards the red cube, the robot arm is in a factory setting, the robot arm is metallic and shiny, the red cube is on a conveyor belt, the robot arm is precise and controlled, the scene is dynamic and industrial"
# negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation"
num_frames = 33
H, W = 480, 832

print(f"🎬 Generating video with torch V-JEPA guidance...")
frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames, height=H, width=W, generator=generator, num_inference_steps=50).frames[0]

export_to_video(frames, f"./guidance_sample_robotarm_torch.mp4", fps=16)  # Export the video

# Test the torch V-JEPA scoring on the generated video
print(" Computing V-JEPA score with torch implementation...")
video_tensor = preprocess_video_for_torch_vjepa(frames)
loss_score = calculate_torch_vjepa_loss(video_tensor, torch_vjepa)
print(f"Torch V-JEPA Loss Score: {loss_score:.4f}")