#!/usr/bin/env python3
"""
Script to find common MP4 videos across directories and create side-by-side GIF comparisons.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Set
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse


def find_common_mp4_files(directories: List[str]) -> Dict[str, List[str]]:
    """
    Find MP4 files that exist in all provided directories.
    
    Args:
        directories: List of directory paths to search
        
    Returns:
        Dictionary mapping common filenames to their full paths in each directory
    """
    if not directories:
        return {}
    
    # Get all MP4 files from each directory
    dir_mp4_files = {}
    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist")
            continue
            
        mp4_files = set()
        for file in os.listdir(dir_path):
            if file.lower().endswith('.mp4'):
                mp4_files.add(file)
        dir_mp4_files[dir_path] = mp4_files
    
    if not dir_mp4_files:
        return {}
    
    # Find common MP4 files across all directories
    common_files = set.intersection(*dir_mp4_files.values())
    
    # Create mapping of common files to their paths
    common_file_paths = {}
    for filename in common_files:
        common_file_paths[filename] = [os.path.join(dir_path, filename) 
                                      for dir_path in dir_mp4_files.keys()]
    
    return common_file_paths


def extract_frames(video_path: str, max_frames: int = None) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract (None for all frames)
        
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames


def create_side_by_side_comparison(video_paths: List[str], k_samples: int = 5, target_height: int = 128) -> List[np.ndarray]:
    """
    Create side-by-side comparison frames from multiple videos.
    
    Args:
        video_paths: List of video file paths
        k_samples: Number of sample frames to extract
        target_height: Target height for all frames
        
    Returns:
        List of side-by-side comparison frames
    """
    # Extract frames from each video
    all_video_frames = []
    for video_path in video_paths:
        frames = extract_frames(video_path, k_samples)
        all_video_frames.append(frames)
    
    # Find the minimum number of frames across all videos
    min_frames = min(len(frames) for frames in all_video_frames)
    
    # Create side-by-side comparisons
    comparison_frames = []
    for i in range(min_frames):
        # Get frame from each video
        frame_list = [frames[i] for frames in all_video_frames]
        
        # Resize all frames to target height while maintaining aspect ratio
        resized_frames = []
        for frame in frame_list:
            h, w = frame.shape[:2]
            new_w = int(w * target_height / h)
            resized = cv2.resize(frame, (new_w, target_height))
            resized_frames.append(resized)
        
        # Calculate total width needed
        total_width = sum(frame.shape[1] for frame in resized_frames)
        total_height = target_height
        
        # Create combined frame
        combined_frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        # Place frames side by side
        x_offset = 0
        for frame in resized_frames:
            combined_frame[:, x_offset:x_offset + frame.shape[1]] = frame
            x_offset += frame.shape[1]
        
        comparison_frames.append(combined_frame)
    
    return comparison_frames


def add_labels_to_frame(frame: np.ndarray, labels: List[str], video_paths: List[str], target_height: int = 128) -> Image.Image:
    """
    Add labels to a frame showing which directory each video comes from.
    
    Args:
        frame: The combined frame as numpy array
        labels: List of labels for each directory
        video_paths: List of video paths to calculate section widths
        target_height: Target height used for frame resizing
        
    Returns:
        PIL Image with labels added
    """
    # Convert to PIL Image
    pil_frame = Image.fromarray(frame)
    
    # Create a copy of the frame to draw labels on
    labeled_frame = pil_frame.copy()
    draw = ImageDraw.Draw(labeled_frame)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Calculate label positions based on actual video section widths
    x_offset = 0
    for i, (label, video_path) in enumerate(zip(labels, video_paths)):
        # Calculate section width based on the first frame of this video
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            h, w = first_frame.shape[:2]
            section_width = int(w * target_height / h)
        else:
            # Fallback: estimate section width
            section_width = frame.shape[1] // len(labels)
        
        # Calculate label position (above the video section)
        label_x = x_offset + section_width // 2
        label_y = 10  # 10 pixels from top
        
        # Center the text horizontally
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = label_x - text_width // 2
        
        # Ensure text stays within the frame bounds
        text_x = max(5, min(text_x, frame.shape[1] - text_width - 5))
        
        # Draw text with black outline for better visibility
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, label_y + dy), label, fill=(0, 0, 0), font=font)
        
        draw.text((text_x, label_y), label, fill=(255, 255, 255), font=font)
        
        x_offset += section_width
    
    return labeled_frame


def save_as_gif(frames: List[np.ndarray], output_path: str, fps: int = 16, labels: List[str] = None, video_paths: List[str] = None, target_height: int = 128):
    """
    Save frames as a GIF file.
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Output GIF file path
        fps: Frames per second for the GIF
        labels: Optional list of labels to add to frames
        video_paths: List of video paths for label positioning
        target_height: Target height used for frame resizing
    """
    # Convert numpy arrays to PIL Images
    pil_images = []
    for frame in frames:
        if labels and video_paths:
            # Add labels to the frame
            pil_image = add_labels_to_frame(frame, labels, video_paths, target_height)
        else:
            pil_image = Image.fromarray(frame)
        pil_images.append(pil_image)
    
    # Calculate duration for each frame
    duration = int(1000 / fps)  # Duration in milliseconds
    
    # Save as GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )


def main():
    # Hardcoded directory list - modify these paths as needed
    # split = "physics_iq"
    # DIRECTORY_LIST = [
    #     f"/home/yjianhao/project/frame-guidance/generated_videos/{split}/Cosmos-Predict2-2B-Video2World/vanilla_v1_f93_s35_cfg7.0",
    #     f"/home/yjianhao/project/frame-guidance/generated_videos/{split}/Cosmos-Predict2-2B-Video2World/guidance_v1_f93_s35_c8_cfg7.0_max", 
    #     f"/home/yjianhao/project/frame-guidance/generated_videos/{split}/Cosmos-Predict2-2B-Video2World/rejection_v1_f93_s35_cfg7.0_reject10_max"
    # ]

    split = "gr1_env"
    DIRECTORY_LIST = [
        f"/home/yjianhao/project/frame-guidance/generated_videos/{split}/Cosmos-Predict2-2B-Video2World/vanilla_v1_f93_s35_cfg7.0",
        f"/home/yjianhao/project/frame-guidance/generated_videos/{split}/Cosmos-Predict2-2B-Video2World/guidance_v1_f93_s35_c8_cfg7.0_max", 
        f"/home/yjianhao/project/frame-guidance/generated_videos/{split}/Cosmos-Predict2-2B-Video2World/rejection_v1_f93_s35_cfg7.0_reject10_max"
    ]
    
    # Labels for each directory (must match the order of DIRECTORY_LIST)
    DIRECTORY_LABELS = [
        "Vanilla",
        "Guidance max", 
        "Rejection 10"
    ]
    
    parser = argparse.ArgumentParser(description='Create side-by-side video comparisons as GIFs')
    parser.add_argument('-o', '--output_dir', default='./visualization/visualize_selected_gr1_env',
                       help='Output directory to save GIF files')
    parser.add_argument('-k', '--samples', type=int, default=93, 
                       help='Number of sample frames to extract (default: 5)')
    parser.add_argument('-f', '--fps', type=int, default=16,
                       help='FPS for the output GIF (default: 16)')
    parser.add_argument('-n', '--num_common', type=int, default=10,
                       help='Number of common MP4s to process/save (0 means all)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Searching for common MP4 files in {len(DIRECTORY_LIST)} directories...")
    print(f"Directories: {', '.join(DIRECTORY_LIST)}")
    print(f"Labels: {', '.join(DIRECTORY_LABELS)}")
    
    # Find common MP4 files
    common_files = find_common_mp4_files(DIRECTORY_LIST)
    
    if not common_files:
        print("No common MP4 files found across all directories.")
        return
    
    print(f"Found {len(common_files)} common MP4 files:")
    for i, filename in enumerate(sorted(common_files)):
        print(f"  {i+1}. {filename}")
    
    # Select subset of common files if requested
    common_items = sorted(common_files.items())
    if args.num_common and args.num_common > 0:
        common_items = common_items[:args.num_common]

    # Process selected common files and create GIFs
    print(f"\nProcessing {len(common_items)} common MP4 files...")
    
    for i, (filename, video_paths) in enumerate(common_items):
        print(f"\n[{i+1}/{len(common_items)}] Processing: {filename}")
        print(f"Found in: {', '.join(video_paths)}")
        
        # Create side-by-side comparison
        print(f"Creating side-by-side comparison with {args.samples} samples...")
        comparison_frames = create_side_by_side_comparison(video_paths, args.samples, target_height=128)
        
        if not comparison_frames:
            print("No frames could be extracted from the videos. Skipping...")
            continue
        
        # Generate output filename (remove .mp4 extension and add .gif)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_comparison.gif"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Save as GIF
        print(f"Saving GIF with {args.fps} FPS...")
        save_as_gif(comparison_frames, output_path, args.fps, DIRECTORY_LABELS, video_paths, target_height=128)
        
        print(f"GIF saved as: {output_path}")
        print(f"Dimensions: {comparison_frames[0].shape[1]}x{comparison_frames[0].shape[0]}")
        print(f"Frames: {len(comparison_frames)}")
    
    print(f"\nAll GIFs have been saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
