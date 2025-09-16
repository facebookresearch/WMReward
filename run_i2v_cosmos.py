# from pipelines.pipeline_cosmos_image2video_14b import Cosmos2VideoToWorldPipeline
# from pipelines.pipeline_cosmos_image2video_savemem import Cosmos2VideoToWorldPipeline
# from pipelines.pipeline_cosmos_image2video_v1 import Cosmos2VideoToWorldPipeline
from pipelines.pipeline_cosmos_image2video_v2 import Cosmos2VideoToWorldPipeline
import os
import torch
from datetime import datetime
from diffusers.utils import export_to_video, load_image, load_video
import math
from decord import VideoReader, cpu
import numpy as np
from PIL import Image


def downsample_video_to_fps(video_path, original_fps, target_fps, num_frames=None):
    """
    Downsample video to exact target FPS using decord.
    
    Args:
        video_path: Path to video file
        original_fps: Original frame rate
        target_fps: Target frame rate
        num_frames: Number of frames to select (if None, selects all available)
    
    Returns:
        List of PIL Image frames at target FPS
    """
    if target_fps >= original_fps:
        # If target FPS is higher, just load all frames
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_indices = list(range(min(total_frames, num_frames) if num_frames else total_frames))
        frames = vr.get_batch(frame_indices).asnumpy()
        
        frame_list = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_frame = Image.fromarray(frame, mode='RGB')
            frame_list.append(pil_frame)
        return frame_list
    
    # Calculate frame interval for exact downsampling
    frame_interval = original_fps / target_fps
    
    # Read video with decord
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    print(f"Original video: {total_frames} frames at {original_fps} fps")
    
    # Calculate frame indices for target FPS
    frame_indices = []
    current_time = 0.0
    
    while current_time < total_frames:
        frame_idx = int(round(current_time))
        if frame_idx < total_frames:
            frame_indices.append(frame_idx)
        current_time += frame_interval
    
    # Limit to requested number of frames
    if num_frames and len(frame_indices) > num_frames:
        frame_indices = frame_indices[-num_frames:]  # Take last N frames
    
    # Read the selected frames
    frames = vr.get_batch(frame_indices).asnumpy()
    
    # Convert frames to PIL Images
    frame_list = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        pil_frame = Image.fromarray(frame, mode='RGB')
        frame_list.append(pil_frame)
    print(f"Downsampled video: {len(frame_list)} frames at exactly {target_fps} fps")
    return frame_list


import argparse
parser = argparse.ArgumentParser("Cosmos I2V with VJEPA slice_pred guidance")
# High-level IO and model
parser.add_argument("--prompt", type=str, default=(
    "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement."
))
parser.add_argument("--init_image", type=str, default="./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg")
parser.add_argument("--init_video", type=str, default=None)
parser.add_argument("--model_id", type=str, default="nvidia/Cosmos-Predict2-2B-Video2World")
parser.add_argument("--num_frames", type=int, default=93)
parser.add_argument("--height", type=int, default=704)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--steps", type=int, default=35)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--guidance_scale", type=float, default=7.0)
# Guidance scheduling
parser.add_argument("--guidance_step_pattern", type=str, default="0x5,1x45")
parser.add_argument("--guidance_lr_pattern", type=str, default="0.005x50")
parser.add_argument("--guidance_frequency", type=int, default=1)
parser.add_argument("--travel_time", type=str, default="0,0")
# VJEPA slice_pred knobs
parser.add_argument("--vjepa_variant", type=str, default="vit_giant", choices=["vit_large","vit_huge","vit_giant","vit_giant_384"])
parser.add_argument("--vjepa_img_size", type=int, default=256)
parser.add_argument("--vjepa_masking_mode", type=str, default="causal", choices=["causal","random"])
parser.add_argument("--vjepa_context_frames", type=int, default=8)
parser.add_argument("--vjepa_mask_ratio", type=float, default=0.75)
parser.add_argument("--slice_window_size", type=int, default=16)
parser.add_argument("--slice_stride", type=int, default=8)
parser.add_argument("--vae_decode_scale", type=float, default=0.3)
parser.add_argument("--loss_mode", type=str, default="max", choices=["mean","max"])
# Output
parser.add_argument("--out_dir", type=str, default="results/cosmos_i2v")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--fps", type=int, default=16)
args = parser.parse_args()

pipe = Cosmos2VideoToWorldPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.enable_gradient_checkpointing()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"

# Build guidance schedules
def build_seq(pattern: str, steps: int, is_float: bool):
    tokens = [p.strip() for p in pattern.split(",") if p.strip()]
    seq = []
    for tok in tokens:
        if "x" in tok:
            v, c = tok.split("x")
            seq.extend(([float(v) if is_float else int(v)]) * int(c))
        else:
            v = float(tok) if is_float else int(tok)
            seq = [v] * steps
            break
    if len(seq) != steps:
        raise ValueError(f"bad pattern len {len(seq)} vs {steps}")
    return seq

def parse_range_pair(text: str):
    a, b = (text.split(",", 1) if "," in text else text.split("-", 1))
    return int(a.strip()), int(b.strip())

os.makedirs(args.out_dir, exist_ok=True)
run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(args.out_dir, f"{run_name}.mp4")

steps = int(args.steps)
guidance_step = build_seq(args.guidance_step_pattern, steps, is_float=False)
guidance_lr = build_seq(args.guidance_lr_pattern, steps, is_float=True)
travel_time = parse_range_pair(args.travel_time)

# Prepare init input
image_input = None
video_input = None
if args.init_image:
    image_input = load_image(args.init_image)
elif args.init_video:
    video_input = load_video(args.init_video)
else:
    raise ValueError("Provide either --init_image or --init_video")

result = pipe(
    image=image_input,
    video=video_input,
    prompt=args.prompt,
    num_frames=args.num_frames,
    height=args.height,
    width=args.width,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=args.guidance_scale,
    generator=torch.Generator(device="cuda").manual_seed(args.seed),
    loss_fn="slice_pred",
    guidance_step=guidance_step,
    guidance_lr=guidance_lr,
    guidance_frequency=args.guidance_frequency,
    additional_inputs={
        "vjepa_variant": args.vjepa_variant,
        "vjepa_img_size": args.vjepa_img_size,
        "vjepa_dtype": "fp32",
        "vjepa_masking_mode": args.vjepa_masking_mode,
        "vjepa_context_frames": args.vjepa_context_frames,
        "vjepa_mask_ratio": args.vjepa_mask_ratio,
        "slice_window_size": args.slice_window_size,
        "slice_stride": args.slice_stride,
        "vae_decode_scale": args.vae_decode_scale,
        "loss_mode": args.loss_mode,
        # Optional intermediate saving controls
        # "save_intermediate": False,
        # "intermediate_save_dir": os.path.join(args.out_dir, f"{run_name}_intermediate"),
        # "intermediate_fps": args.fps,
    },
    fps=args.fps,
    travel_time=travel_time,
).frames[0]

export_to_video(result, out_path, fps=args.fps)
print("Saved:", out_path)
