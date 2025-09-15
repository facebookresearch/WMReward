# from pipelines.pipeline_cosmos_image2video_14b import Cosmos2VideoToWorldPipeline
# from pipelines.pipeline_cosmos_image2video_savemem import Cosmos2VideoToWorldPipeline
# from pipelines.pipeline_cosmos_image2video_v1 import Cosmos2VideoToWorldPipeline
from pipelines.pipeline_cosmos_image2video_v2 import Cosmos2VideoToWorldPipeline
import torch
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
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='guidance', choices=['guidance', 'vanilla'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--guidance_step_pattern', type=str, default="0x3,2x12,3x10,1x10")
parser.add_argument('--guidance_lr_pattern', type=str, default="0.003x35")
parser.add_argument('--loss_mode', type=str, default='max', choices=['mean', 'max'])
parser.add_argument('--cfg', type=float, default=7.0)
parser.add_argument('--k_frames', type=int, default=5, help='Number of final frames to select after downsampling')
parser.add_argument('--guidance_freq', type=int, default=1)
args = parser.parse_args()

model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
# model_id = "nvidia/Cosmos-Predict2-14B-Video2World"
pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
pipe = pipe.to("cuda")
pipe.transformer.enable_gradient_checkpointing()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
# pipe.vae.enable_gradient_checkpointing()

# prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
# prompt = "A robot arm performing precise manipulation task and reaching for the drawer, smooth realistic motion"
# negative_prompt = "The video captures a series of frames showing ugly scenes, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png"
# )
# image = load_image("/home/yjianhao/project/video_guidance/droid/8756/original_buffer/frame_0000.png")
# image = load_image("/home/yjianhao/project/frame-guidance/dream_gen_benchmark/gr1_object/0_Use the left hand to pick up dark green cucumber from on circular gray mat to above beige bowl..png")
# prompt = "Use the left hand to pick up dark green cucumber from on circular gray mat to above beige bowl."


# image = load_image("/home/yjianhao/project/frame-guidance/dream_gen_benchmark/gr1_object/1_Use the right hand to pick up orange juice carton from center of pink plate to center of green bowl..png")
# prompt = "Use the right hand to pick up orange juice carton from center of pink plate to center of green bowl."

# image = load_image("/home/yjianhao/project/frame-guidance/dream_gen_benchmark/gr1_object/4_Use the right hand to pick up orange carrot from center of table to lower white shelf..png")
# prompt="Use the right hand to pick up orange carrot from center of table to lower white shelf."
# negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"

# image = load_image("/home/yjianhao/project/frame-guidance/dream_gen_benchmark/gr1_object/4_Use the right hand to pick up orange carrot from center of table to lower white shelf..png")
# video_path = "/home/yjianhao/project/PhysicsIQ/physics-IQ-benchmark/split-videos/conditioning/30FPS/0001_conditioning-videos_30FPS_perspective-left_take-1_trimmed-ball-and-block-fall.mp4"

# # Use decord to downsample video to exact 16 fps and select final k frames
# video = downsample_video_to_fps(video_path, original_fps=30, target_fps=16, num_frames=args.k_frames)
# print(f"Processed video: {len(video)} frames at exactly 16 fps, size: {video[0].size}")
# export_to_video(video, "temp.mp4", fps=5)
# print(f"Saved conditioned input video as temp.mp4")

prompt="Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement."
negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"

image = load_image("/home/yjianhao/project/PhysicsIQ/physics-IQ-benchmark/switch-frames/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg")

# Build guidance schedules like in CogVideoX script
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

steps = 35

if args.mode == 'guidance':
#   guidance_step = build_seq("0x5,10x15,5x15", steps, is_float=False)  # 0 for first 3, 3 for next 12, 2 for rest
  guidance_step = build_seq(args.guidance_step_pattern, steps, is_float=False)  # 0 for first 3, 3 for next 12, 2 for rest
  # guidance_step = build_seq("0x3,1x12,1x10,1x10", steps, is_float=False)  # 0 for first 3, 3 for next 12, 2 for rest
  # guidance_step = build_seq("0x3,3x10,2x10,1x12", steps, is_float=False)  # 0 for first 3, 3 for next 12, 2 for rest
  # guidance_lr = build_seq("3.0x15,2.0x20", steps, is_float=True)     # 3.0 for first 15, 2.0 for rest
  # guidance_lr = build_seq("30x15,20x15,10x5", steps, is_float=True)     # 3.0 for first 15, 2.0 for rest
  guidance_lr = build_seq(args.guidance_lr_pattern, steps, is_float=True)     # 3.0 for first 15, 2.0 for rest
  # guidance_step = build_seq("0x35", steps, is_float=False)  # 0 for first 3, 3 for next 12, 2 for rest
  # guidance_lr = build_seq("0x35", steps, is_float=True)     # 3.0 for first 15, 2.0 for rest
  travel_time = (0, 0)  # time travel window
elif args.mode == 'vanilla':
  guidance_step = build_seq("0x35", steps, is_float=False)  # 0 for first 3, 3 for next 12, 2 for rest
  guidance_lr = build_seq("0x35", steps, is_float=True)     # 3.0 for first 15, 2.0 for rest
  travel_time = (0, 0)  # time travel window

video = pipe(
  image=image,
#   video=video,
  prompt=prompt,
  num_frames=93,
  height=704,
  width=1280,
  negative_prompt=negative_prompt,
  num_inference_steps=steps,
  guidance_scale=args.cfg,
  generator=torch.Generator(device="cuda").manual_seed(args.seed),
  loss_fn="slice_pred",
  guidance_step=guidance_step,
  guidance_lr=guidance_lr,
  guidance_frequency=args.guidance_freq,
  additional_inputs={
    "vjepa_variant": "vit_giant",
    "vjepa_img_size": 256,
    "vjepa_dtype": "fp32",
    "vjepa_masking_mode": "causal",
    "vjepa_context_frames": 8,
    "vjepa_mask_ratio": 0.75,
    "slice_window_size": 16,
    "slice_stride": 8,
    "vae_decode_scale": 0.3,
    "loss_mode": args.loss_mode,
    "save_intermediate": False,  # Enable/disable intermediate saving
    "intermediate_save_dir": f"./results/cosmos_savemem_{args.lr}_{args.mode}_{args.loss_mode}_{args.cfg}_{args.seed}",  # Custom save directory
    "intermediate_fps": 16,  # FPS for intermediate videos
  },
  fps=16,
  travel_time=travel_time,
).frames[0]
if args.mode == 'guidance':
  export_to_video(video, f"cosmos2b_{args.guidance_lr_pattern}_{args.guidance_step_pattern}_{args.guidance_freq}.mp4", fps=16)
else:
  export_to_video(video, f"cosmos2b_vanilla_{args.cfg}_{args.seed}_{args.guidance_freq}.mp4", fps=16)
