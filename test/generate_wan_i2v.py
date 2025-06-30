import torch
import numpy as np
from diffusers import AutoencoderKLWan
from pipelines.wan_i2v_pipeline import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
import gpustat

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
# )
# max_area = 480 * 832
# aspect_ratio = image.height / image.width
# mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
# height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
# width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = load_image("/home/yjianhao/project/video_guidance/temp/sample.png")
height, width = 480, 832
image = image.resize((width, height))
prompt = (
    "The gripper moving forward to pickup the ball"
    "fixed camera angle"
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
print(height, width)
output = pipe(
    image=image, prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, num_frames=33, guidance_scale=5.0
).frames[0]
export_to_video(output, "output_wan.mp4", fps=16)
