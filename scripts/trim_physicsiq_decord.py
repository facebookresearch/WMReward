#!/usr/bin/env python3

import os
import argparse
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu
from diffusers.utils import export_to_video
from PIL import Image

def trim_video_to_frames(input_path, output_path, target_fps=8, target_frames=40, target_size=None):
    """Trim a video to specified number of frames at target FPS using decord and diffusers."""
    # Read video with decord
    vr = VideoReader(input_path, ctx=cpu(0))
    
    # Get the first N frames (sequential trimming, not subsampling)
    total_frames = len(vr)
    frames_to_take = min(target_frames, total_frames)
    
    # Read the first N frames sequentially
    frame_indices = list(range(frames_to_take))
    frames = vr.get_batch(frame_indices).asnumpy()
    
    # Convert frames to PIL Images for proper color handling
    frame_list = []
    for i in range(len(frames)):
        # Ensure frame is uint8 and in RGB format
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        # Convert to PIL Image to ensure proper color handling
        pil_frame = Image.fromarray(frame, mode='RGB')
        
        # Resize frame if target size is specified
        if target_size:
            pil_frame = pil_frame.resize(target_size, Image.Resampling.LANCZOS)
        
        frame_list.append(pil_frame)
    
    # Use diffusers export_to_video for proper MP4 encoding
    export_to_video(frame_list, output_path, fps=target_fps)
    
    size_info = f" {target_size}" if target_size else ""
    print(f"✓ Trimmed: {os.path.basename(input_path)} ({frames_to_take} frames{size_info})")
    return True

def trim_all_videos_in_folder(input_folder, output_folder=None, target_fps=8, target_frames=40, target_size=None):
    """Trim all .mp4 videos in a folder to specified frames at target FPS."""
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
            if trim_video_to_frames(str(video_file), str(output_path), target_fps, target_frames, target_size):
                success_count += 1
        except Exception as e:
            print(f"✗ Failed to process {video_file.name}: {e}")
    
    print(f"\nCompleted: {success_count}/{len(video_files)} videos trimmed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim videos to specified frames and FPS for Physics-IQ benchmark")
    parser.add_argument("--input_folders", nargs="+", default=["/home/yjianhao/project/frame-guidance/generated_videos/physics_iq/Cosmos-Predict2-2B-Video2World/guidance_v1_f93_s35_c8_cfg7.0_max"], help="List of folders containing videos to trim")
    # parser.add_argument("--output_folder", default="njfknjkliutuhcnccktildjjcetjjjkd/home/yjianhao/project/frame-guidance/generated_videos/physics_iq/Cosmos-Predict2-2B-Video2World/guidance_v1_f93_s35_c8_cfg7.0_max_8trimmed", help="Output folder (default: input_folder/trimmed)")
    parser.add_argument("--target_fps", type=int, default=8, help="Target FPS for output videos (default: 8)")
    parser.add_argument("--target_frames", type=int, default=49, help="Target number of frames for output videos (default: 40)")
    parser.add_argument("--target_width", type=int, default=1280, help="Target width for output videos (default: 720)")
    parser.add_argument("--target_height", type=int, default=704, help="Target height for output videos (default: 480)")
    # parser.add_argument("--target_width", type=int, default=720, help="Target width for output videos (default: 720)")
    # parser.add_argument("--target_height", type=int, default=480, help="Target height for output videos (default: 480)")
    # parser.add_argument("--target_fps", type=int, default=16, help="Target FPS for output videos (default: 8)")
    # parser.add_argument("--target_frames", type=int, default=80, help="Target number of frames for output videos (default: 40)")
    
    args = parser.parse_args()
    
    # Create target_size tuple from width and height
    target_size = (args.target_width, args.target_height)
    
    # Process each input folder
    for input_folder in args.input_folders:
        if not os.path.exists(input_folder):
            print(f"Error: Input folder does not exist: {input_folder}")
            continue
        
        output_folder = input_folder + "_8trimmed"
        print(f"\nProcessing folder: {input_folder}")
        trim_all_videos_in_folder(input_folder, output_folder, args.target_fps, args.target_frames, target_size)
