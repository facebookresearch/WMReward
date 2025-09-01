#!/usr/bin/env python3

import os
import argparse
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu
from diffusers.utils import export_to_video
from PIL import Image

def trim_video_to_frames(input_path, output_path, target_fps=8, target_frames=40, original_fps=None, original_frames=None):
    """Subsample a video to specified number of frames at target FPS using decord and diffusers."""
    # Read video with decord
    vr = VideoReader(input_path, ctx=cpu(0))
    
    # Use provided original values or detect from video
    if original_fps is None:
        original_fps = vr.get_avg_fps()
    if original_frames is None:
        original_frames = len(vr)
    
    # Calculate frame indices to sample
    if target_frames >= original_frames:
        # If target frames >= original frames, use all frames
        frame_indices = list(range(original_frames))
    else:
        # Sample frames evenly across the video
        step = original_frames / target_frames
        frame_indices = [int(i * step) for i in range(target_frames)]
    
    # Read the sampled frames
    frames = vr.get_batch(frame_indices).asnumpy()
    
    # Convert frames to PIL Images (export_to_video expects PIL Images or proper numpy format)
    frame_list = []
    for i in range(len(frames)):
        # Ensure frame is uint8 and in RGB format
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        # Convert to PIL Image to ensure proper color handling
        pil_frame = Image.fromarray(frame, mode='RGB')
        frame_list.append(pil_frame)
    
    # Use diffusers export_to_video for proper MP4 encoding
    export_to_video(frame_list, output_path, fps=target_fps)
    
    actual_frames = len(frame_indices)
    step_used = original_frames / actual_frames if actual_frames > 0 else 1
    
    print(f"✓ Subsampled: {os.path.basename(input_path)} ({actual_frames} frames, step={step_used:.2f})")
    return True

def trim_all_videos_in_folder(input_folder, output_folder=None, target_fps=8, target_frames=40, original_fps=None, original_frames=None):
    """Subsample all .mp4 videos in a folder to specified frames at target FPS."""
    input_folder = Path(input_folder)
    
    if output_folder is None:
        output_folder = input_folder / "trimmed"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(exist_ok=True)
    
    video_files = list(input_folder.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} videos to trim")
    print(f"Output folder: {output_folder}")
    
    success_count = 0
    for video_file in sorted(video_files):
        output_path = output_folder / video_file.name
        try:
            if trim_video_to_frames(str(video_file), str(output_path), target_fps, target_frames, original_fps, original_frames):
                success_count += 1
        except Exception as e:
            print(f"✗ Failed to process {video_file.name}: {e}")
    
    print(f"\nCompleted: {success_count}/{len(video_files)} videos trimmed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample videos to specified frames and FPS for Physics-IQ benchmark")
    parser.add_argument("--input_folder", default="/home/yjianhao/project/frame-guidance/generated_videos/gr1_object/Cosmos-Predict2-14B-Video2World/vanilla_v1_f93_s35_cfg7.0_totrim", help="Folder containing videos to trim")
    parser.add_argument("--output_folder", default="/home/yjianhao/project/frame-guidance/generated_videos/gr1_object/Cosmos-Predict2-14B-Video2World/vanilla_v1_f93_s35_cfg7.0_subsampled", help="Output folder (default: input_folder/trimmed)")
    parser.add_argument("--target_fps", type=int, default=8, help="Target FPS for output videos (default: 8)")
    parser.add_argument("--target_frames", type=int, default=40, help="Target number of frames for output videos (default: 40)")
    parser.add_argument("--original_fps", type=float, default=16, help="Original FPS of input videos (optional, will auto-detect if not provided)")
    parser.add_argument("--original_frames", type=int, default=93, help="Original frame count of input videos (optional, will auto-detect if not provided)")
    # parser.add_argument("--target_fps", type=int, default=16, help="Target FPS for output videos (default: 8)")
    # parser.add_argument("--target_frames", type=int, default=80, help="Target number of frames for output videos (default: 40)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        exit(1)
    
    trim_all_videos_in_folder(args.input_folder, args.output_folder, args.target_fps, args.target_frames, args.original_fps, args.original_frames)
