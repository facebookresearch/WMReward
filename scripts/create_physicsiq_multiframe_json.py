#!/usr/bin/env python3

import json
import os
from pathlib import Path

def create_multiframe_json():
    """Create physics_iq_multiframe.json with MP4 videos as input instead of single images."""
    
    # Load the existing physics_iq.json to get the structure
    physics_iq_file = "./prompts/physics_iq.json"
    with open(physics_iq_file, 'r') as f:
        physics_iq_data = json.load(f)
    
    # Base directory for the 30FPS conditioning videos
    base_video_dir = "/home/yjianhao/project/PhysicsIQ/physics-IQ-benchmark/split-videos/conditioning/30FPS"
    
    # Output directory for generated videos
    output_base_dir = "output/physics_iq/cosmos_multiframe"
    
    # Generate the JSON data by transforming existing entries
    json_data = []
    
    for entry in physics_iq_data:
        # Extract the scenario name from the original input path
        original_input = entry["input_video"]
        filename = Path(original_input).name  # Get filename with extension
        
        print(f"Processing: {filename}")
        
        # Parse the filename to extract perspective and scenario
        # Format: 0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg
        parts = filename.split('_')
        print(f"  Parts: {parts}")
        
        # Find perspective
        perspective = None
        for part in parts:
            if part.startswith('perspective-'):
                perspective = part
                break
        
        if not perspective:
            perspective = "perspective-center"  # Default fallback
        
        print(f"  Perspective: {perspective}")
        
        # Find the scenario (everything after "trimmed-")
        scenario = None
        for part in parts:
            if part.startswith("trimmed-"):
                # Extract everything after "trimmed-" and remove .jpg extension
                scenario = part[8:].replace('.jpg', '')  # 8 = len("trimmed-")
                break
        
        if not scenario:
            print(f"  Warning: Could not parse scenario from {filename}")
            continue
        
        print(f"  Scenario: {scenario}")
        
        # Find the corresponding video file by searching for the scenario and perspective
        # The video files have different numbering, so we need to search by content
        video_found = False
        for video_file in os.listdir(base_video_dir):
            if video_file.endswith('.mp4') and f"_{perspective}_" in video_file and f"trimmed-{scenario}.mp4" in video_file:
                input_video = os.path.join(base_video_dir, video_file)
                video_found = True
                break
        
        if not video_found:
            print(f"  Warning: Could not find video for {scenario} with {perspective}")
            continue
        
        print(f"  Found video: {os.path.basename(input_video)}")
        
        # Construct the output video path
        output_video = f"{output_base_dir}/0001_trimmed-{scenario}_{perspective}.mp4"
        
        # Create the new entry
        new_entry = {
            "input_video": input_video,
            "prompt": entry["prompt"],
            "output_video": output_video
        }
        
        json_data.append(new_entry)
        print(f"  Added entry for {scenario}")
        print()
    
    # Write to JSON file
    output_file = "./prompts/physics_iq_multiframe.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Generated {len(json_data)} entries in {output_file}")
    print(f"Source entries: {len(physics_iq_data)}")
    print(f"Transformed from: {physics_iq_file}")

if __name__ == "__main__":
    create_multiframe_json()
