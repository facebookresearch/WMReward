#!/usr/bin/env python3

import json
import os
from pathlib import Path
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video

def downsample_video_to_fps(video_path, original_fps, target_fps, num_frames=None):
    """
    Downsample video to exact target FPS using decord.
    Same function as in run_i2v_cosmos.py
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

def process_physics_iq_videos():
    """Process all Physics-IQ videos to extract 5 frames and save them."""
    
    # Load the existing physics_iq.json
    physics_iq_file = "./prompts/physics_iq.json"
    with open(physics_iq_file, 'r') as f:
        physics_iq_data = json.load(f)
    
    # Output directory for 5-frame videos
    output_dir = "/home/yjianhao/project/frame-guidance/physicsiq_5frame"
    os.makedirs(output_dir, exist_ok=True)
    
    # New JSON data
    new_json_data = []
    
    print(f"Processing {len(physics_iq_data)} videos...")
    
    for i, entry in enumerate(physics_iq_data):
        # Get the original input video path
        original_input = entry["input_video"]
        filename = Path(original_input).name  # Get filename with extension
        
        # Parse the filename to extract perspective and scenario
        # Format: 0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg
        parts = filename.split('_')
        
        # Find perspective
        perspective = None
        for part in parts:
            if part.startswith('perspective-'):
                perspective = part
                break
        
        if not perspective:
            perspective = "perspective-center"  # Default fallback
        
        # Find the scenario (everything after "trimmed-")
        scenario = None
        for part in parts:
            if part.startswith("trimmed-"):
                # Extract everything after "trimmed-" and remove .jpg extension
                scenario = part[8:].replace('.jpg', '')  # 8 = len("trimmed-")
                break
        
        if not scenario:
            print(f"Warning: Could not parse scenario from {filename}")
            continue
        
        # Find the corresponding video file by searching for the scenario and perspective
        # The video files have different numbering, so we need to search by content
        base_video_dir = "/home/yjianhao/project/PhysicsIQ/physics-IQ-benchmark/split-videos/conditioning/30FPS"
        video_found = False
        video_path = None
        
        for video_file in os.listdir(base_video_dir):
            if video_file.endswith('.mp4') and f"_{perspective}_" in video_file and f"trimmed-{scenario}.mp4" in video_file:
                video_path = os.path.join(base_video_dir, video_file)
                video_found = True
                break
        
        if not video_found:
            print(f"Warning: Could not find video for {scenario} with {perspective}")
            continue
        
        print(f"Processing {i+1}/{len(physics_iq_data)}: {scenario}")
        
        try:
            # Extract 5 frames using the same method as run_i2v_cosmos.py
            frames = downsample_video_to_fps(video_path, original_fps=30, target_fps=16, num_frames=5)
            
            # Save the 5-frame video
            output_filename = f"0001_5frame_30FPS_{perspective}_take-1_trimmed-{scenario}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as 5 fps (since we have 5 frames)
            export_to_video(frames, output_path, fps=5)
            print(f"  Saved: {output_filename}")
            
            # Create new JSON entry
            new_entry = {
                "input_video": output_path,
                "prompt": entry["prompt"],
                "output_video": entry["output_video"]  # Keep the same output path
            }
            new_json_data.append(new_entry)
            
        except Exception as e:
            print(f"  Error processing {scenario_name}: {e}")
            continue
    
    # Save the new JSON
    new_json_file = "./prompts/physics_iq_5frame.json"
    with open(new_json_file, 'w') as f:
        json.dump(new_json_data, f, indent=4)
    
    print(f"\nCompleted!")
    print(f"Processed videos: {len(new_json_data)}")
    print(f"5-frame videos saved to: {output_dir}")
    print(f"New JSON saved to: {new_json_file}")

if __name__ == "__main__":
    process_physics_iq_videos()
