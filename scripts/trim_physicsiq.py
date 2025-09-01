#!/usr/bin/env python3

import os
import cv2
import argparse
from pathlib import Path

def trim_video_to_frames(input_path, output_path, target_fps=8, target_frames=40):
    """Trim a video to specified number of frames at target FPS using OpenCV."""
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter with target FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_count = 0
    
    try:
        while frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"✓ Trimmed: {os.path.basename(input_path)} ({frame_count} frames)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to trim {input_path}: {e}")
        cap.release()
        out.release()
        return False

def trim_all_videos_in_folder(input_folder, output_folder=None, target_fps=8, target_frames=40):
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
        if trim_video_to_frames(str(video_file), str(output_path), target_fps, target_frames):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(video_files)} videos trimmed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim videos to specified frames and FPS for Physics-IQ benchmark")
    parser.add_argument("--input_folder", default="/home/yjianhao/project/frame-guidance/generated_videos/physics_iq/Cosmos-Predict2-2B-Video2World/guidance_v1_f93_s35_c8_cfg7.0_max", help="Folder containing videos to trim")
    parser.add_argument("--output_folder", default="/home/yjianhao/project/frame-guidance/generated_videos/physics_iq/Cosmos-Predict2-2B-Video2World/guidance_v1_f93_s35_c8_cfg7.0_max_16trimmed", help="Output folder (default: input_folder/trimmed)")
    # parser.add_argument("--target_fps", type=int, default=8, help="Target FPS for output videos (default: 8)")
    # parser.add_argument("--target_frames", type=int, default=40, help="Target number of frames for output videos (default: 40)")
    parser.add_argument("--target_fps", type=int, default=16, help="Target FPS for output videos (default: 8)")
    parser.add_argument("--target_frames", type=int, default=80, help="Target number of frames for output videos (default: 40)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        exit(1)
    
    trim_all_videos_in_folder(args.input_folder, args.output_folder, args.target_fps, args.target_frames)
