#!/usr/bin/env python3
"""
Video Comparison Script

This script combines videos from different folders that correspond to the same prompt
into a single comparison video with labels for easy comparison.
"""

import os
import subprocess
import glob
from pathlib import Path
import tempfile
import shutil

# Configuration
BASE_DIR = "/home/yjianhao/project/video_guidance/generated_videos/subject_consistency"
OUTPUT_DIR = "/home/yjianhao/project/video_guidance/comparison_videos"

# Define the folders you want to compare
FOLDERS_TO_COMPARE = [
    "guidance_f33_s50_w16c8_rho100.0_cfg5.0torch",
    "guidance_f33_s50_w16c8_rho50.0_cfg5.0torch", 

    "guidance_f33_s50_w16c8_rho10.0_cfg5.0torch",
    "guidance_f33_s50_w16c8_rho3.0_cfg5.0torch",
    "guidance_f33_s50_w16c8_rho1.0_cfg5.0torch",
    
    "vanilla_f33_s50_cfg5.0"
]

# Define shorter labels for each folder (for display in video)
FOLDER_LABELS = {
    "guidance_f33_s50_w16c8_rho100.0_cfg5.0torch": "Guidance rho=100",
    "guidance_f33_s50_w16c8_rho50.0_cfg5.0torch": "Guidance rho=50", 
    "guidance_f33_s50_w16c8_rho3.0_cfg5.0torch": "Guidance rho=3",
    "guidance_f33_s50_w16c8_rho1.0_cfg5.0torch": "Guidance rho=1",
    "guidance_f33_s50_w16c8_rho10.0_cfg5.0torch": "Guidance rho=10",
    # "rejection_f33_s50_w8c6_a10_cfg5.0": "Rejection",
    "vanilla_f33_s50_cfg5.0": "Vanilla"
}

# Define specific prompts you want to compare (or leave empty to process all)
SPECIFIC_PROMPTS = [
    "a cat running happily.mp4",
    "a motorcycle slowing down to stop.mp4",
    "a boat sailing smoothly on a calm lake.mp4",
    "a horse taking a peaceful walk.mp4",
    "a bicycle gliding through a snowy field.mp4",
    "a person playing guitar.mp4",
    "a dog running happily.mp4",
    "a car accelerating to gain speed.mp4",
    "a bear catching a salmon in its powerful jaws.mp4"
]

def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def get_available_prompts():
    """Get all available prompts from the first folder."""
    first_folder = os.path.join(BASE_DIR, FOLDERS_TO_COMPARE[0])
    if not os.path.exists(first_folder):
        print(f"Error: Folder {first_folder} does not exist")
        return []
    
    video_files = glob.glob(os.path.join(first_folder, "*.mp4"))
    prompts = [os.path.basename(f) for f in video_files]
    return sorted(prompts)

def find_videos_for_prompt(prompt):
    """Find all videos for a given prompt across different folders."""
    videos = {}
    for folder in FOLDERS_TO_COMPARE:
        video_path = os.path.join(BASE_DIR, folder, prompt)
        if os.path.exists(video_path):
            videos[folder] = video_path
        else:
            print(f"Warning: {prompt} not found in {folder}")
    return videos

def add_text_overlay(input_video, output_video, text, position="top"):
    """Add text overlay to a video using FFmpeg."""
    # Choose position
    if position == "top":
        drawtext_pos = "x=(w-text_w)/2:y=30"
    elif position == "bottom":
        drawtext_pos = "x=(w-text_w)/2:y=h-60"
    else:
        drawtext_pos = "x=(w-text_w)/2:y=30"
    
    cmd = [
        "ffmpeg", "-i", input_video,
        "-vf", f"drawtext=text='{text}':fontsize=24:fontcolor=white:box=1:boxcolor=black@0.8:boxborderw=5:{drawtext_pos}",
        "-c:a", "copy",
        "-y", output_video
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding text overlay: {e}")
        return False

def combine_videos_grid(video_paths, labels, output_path, prompt_name):
    """Combine videos in a grid layout with labels."""
    if len(video_paths) == 0:
        return False
    
    # Create temporary directory for processed videos
    with tempfile.TemporaryDirectory() as temp_dir:
        labeled_videos = []
        
        # Add labels to each video
        for i, (folder, video_path) in enumerate(video_paths.items()):
            label = labels.get(folder, folder)
            temp_video = os.path.join(temp_dir, f"labeled_{i}.mp4")
            
            if add_text_overlay(video_path, temp_video, label):
                labeled_videos.append(temp_video)
            else:
                print(f"Failed to add label to {video_path}")
                return False
        
        # Determine grid layout based on number of videos
        num_videos = len(labeled_videos)
        if num_videos == 1:
            # Just copy the labeled video
            shutil.copy2(labeled_videos[0], output_path)
            return True
        elif num_videos == 2:
            layout = "hstack=inputs=2"
        elif num_videos <= 4:
            layout = "hstack=inputs=2" if num_videos == 2 else "vstack=inputs=2[v1];[v1]hstack=inputs=2" if num_videos == 3 else "hstack=inputs=2[top];hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2"
        elif num_videos <= 6:
            layout = "hstack=inputs=3[top];hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2"
        else:
            # For more than 6 videos, we'll do a 3x3 grid or similar
            layout = "hstack=inputs=3[row1];hstack=inputs=3[row2];hstack=inputs=3[row3];[row1][row2][row3]vstack=inputs=3"
        
        # Build FFmpeg command for combining videos
        cmd = ["ffmpeg"]
        
        # Add input videos
        for video in labeled_videos:
            cmd.extend(["-i", video])
        
        # Add filter complex for grid layout
        if num_videos == 2:
            cmd.extend(["-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]"])
        elif num_videos == 3:
            cmd.extend(["-filter_complex", "[0:v][1:v]hstack=inputs=2[top];[2:v]pad=2*iw:ih[bottom];[top][bottom]vstack=inputs=2[v]", "-map", "[v]"])
        elif num_videos == 4:
            cmd.extend(["-filter_complex", "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]", "-map", "[v]"])
        elif num_videos == 5:
            cmd.extend(["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v]hstack=inputs=2[bottom_partial];[bottom_partial]pad=1.5*iw:ih[bottom];[top][bottom]vstack=inputs=2[v]", "-map", "[v]"])
        elif num_videos == 6:
            cmd.extend(["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[v]", "-map", "[v]"])
        else:
            # Fallback: just stack the first 6 videos
            cmd.extend(["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[top];[3:v][4:v][5:v]hstack=inputs=3[bottom];[top][bottom]vstack=inputs=2[v]", "-map", "[v]"])
        
        # Add output settings
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-y", output_path
        ])
        
        print(f"Combining {num_videos} videos for prompt: {prompt_name}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error combining videos: {e}")
            return False

def process_prompts(prompts):
    """Process a list of prompts and create comparison videos."""
    successful = 0
    failed = 0
    
    for prompt in prompts:
        print(f"\nProcessing: {prompt}")
        
        # Find videos for this prompt
        videos = find_videos_for_prompt(prompt)
        
        if len(videos) < 2:
            print(f"Skipping {prompt}: found only {len(videos)} video(s)")
            failed += 1
            continue
        
        # Create output filename
        prompt_name = os.path.splitext(prompt)[0]
        safe_name = "".join(c for c in prompt_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_file = os.path.join(OUTPUT_DIR, f"comparison_{safe_name}.mp4")
        
        # Combine videos
        if combine_videos_grid(videos, FOLDER_LABELS, output_file, prompt_name):
            print(f"✓ Created: {output_file}")
            successful += 1
        else:
            print(f"✗ Failed to create comparison for: {prompt}")
            failed += 1
    
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")

def main():
    """Main function."""
    print("Video Comparison Script")
    print("=" * 50)
    
    # Create output directory
    create_output_dir()
    
    # Check if specific prompts are defined
    if SPECIFIC_PROMPTS:
        prompts_to_process = SPECIFIC_PROMPTS
        print(f"Processing {len(prompts_to_process)} specific prompts")
    else:
        # Get all available prompts
        prompts_to_process = get_available_prompts()
        print(f"Found {len(prompts_to_process)} prompts to process")
    
    if not prompts_to_process:
        print("No prompts found to process!")
        return
    
    # Process prompts
    process_prompts(prompts_to_process)

if __name__ == "__main__":
    main() 