import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from typing import List, Tuple

def extract_frames(video_path: str, max_frames: int = None) -> List[np.ndarray]:
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames

def create_grid_frame(video_frames: List[np.ndarray], labels: List[str], 
                     target_height: int = 128, ncol: int = 2) -> np.ndarray:
    """Create a single frame with videos arranged in a grid"""
    if not video_frames:
        return None
    
    # Calculate number of rows needed
    nrow = (len(video_frames) + ncol - 1) // ncol
    
    # Resize all frames to target height
    resized_frames = []
    for frame in video_frames:
        h, w = frame.shape[:2]
        new_w = int(w * target_height / h)
        resized = cv2.resize(frame, (new_w, target_height))
        resized_frames.append(resized)
    
    # Calculate total dimensions
    max_width = max(frame.shape[1] for frame in resized_frames)
    total_width = max_width * ncol
    total_height = target_height * nrow
    
    # Create grid layout
    combined_frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    for i, frame in enumerate(resized_frames):
        row = i // ncol
        col = i % ncol
        
        # Center frame in its grid cell
        y_start = row * target_height
        x_start = col * max_width + (max_width - frame.shape[1]) // 2
        
        combined_frame[y_start:y_start + target_height, 
                      x_start:x_start + frame.shape[1]] = frame
        
        # Debug: print the layout order
        print(f"Video {i}: Row {row}, Col {col} - Position: ({x_start}, {y_start})")
    
    return combined_frame

def add_labels_to_frame(frame: np.ndarray, labels: List[str], 
                       target_height: int = 128, ncol: int = 2) -> Image.Image:
    """Add labels inside each video frame near the bottom edge"""
    # Convert to PIL Image
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Create a copy of the frame to draw labels on
    labeled_frame = pil_frame.copy()
    draw = ImageDraw.Draw(labeled_frame)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Calculate grid cell dimensions
    cell_width = frame.shape[1] // ncol
    cell_height = target_height
    
    for i, label in enumerate(labels):
        # Calculate grid position
        row = i // ncol
        col = i % ncol
        
        # Calculate label position (inside the video frame, near bottom edge)
        label_x = col * cell_width + cell_width // 2
        label_y = row * cell_height + target_height - 20  # 20 pixels from bottom
        
        # Center the text horizontally
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = label_x - text_width // 2
        
        # Ensure text stays within the video frame bounds
        text_x = max(5, min(text_x, frame.shape[1] - text_width - 5))
        
        # Draw text with black outline for better visibility
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, label_y + dy), label, fill=(0, 0, 0), font=font)
        
        draw.text((text_x, label_y), label, fill=(255, 255, 255), font=font)
    
    return labeled_frame

def create_grid_gif(video_paths: List[str], labels: List[str], 
                    output_path: str, fps: int = 16, max_frames: int = None, ncol: int = 2) -> None:
    """Create a grid GIF from multiple videos"""
    
    # Extract frames from all videos
    all_video_frames = []
    for video_path in video_paths:
        frames = extract_frames(video_path, max_frames)
        all_video_frames.append(frames)
    
    # Find the minimum number of frames across all videos
    min_frames = min(len(frames) for frames in all_video_frames)
    
    # Create frames for the GIF
    gif_frames = []
    
    for frame_idx in range(min_frames):
        # Get current frame from each video
        current_frames = [frames[frame_idx] for frames in all_video_frames]
        
        # Create grid frame
        combined_frame = create_grid_frame(current_frames, labels, target_height=128, ncol=ncol)
        
        # Add labels
        labeled_frame = add_labels_to_frame(combined_frame, labels, target_height=128, ncol=ncol)
        
        gif_frames.append(labeled_frame)
    
    # Save as GIF
    if gif_frames:
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=1000 // fps,  # Duration in milliseconds
            loop=0
        )
        print(f"GIF saved to: {output_path}")
        print(f"Total frames: {len(gif_frames)}")
        print(f"FPS: {fps}")
        print(f"Grid layout: {ncol} columns")
    else:
        print("No frames to save")

def main():
    # Hard-coded video list and labels
    video_paths = [
        "cosmos2b_vanilla_42.mp4",
        "cosmos2b_lr_0.001_rep_testing7_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_0.005_rep_testing7_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_0.05_rep_testing7_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_10.0_rep_testing6_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_10.0_testing6_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_15.0_rep_testing6_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_30.0_rep_testing6_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_30.0_testing6_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_5.0_rep_testing6_guidance_max_7.0_42.mp4",
        "cosmos2b_lr_5.0_testing6_guidance_max_7.0_42.mp4"
    ]



    labels = [label.replace('.mp4', '') for label in video_paths]
    labels = [label.replace('cosmos2b', '') for label in labels]
    labels = [label.replace('_', ' ') for label in labels]
    labels = [label.replace('guidance ', ' ') for label in labels]

    video_paths = [os.path.join("/home/yjianhao/project/frame-guidance/cosmos2b_orange_42_day2/", v) for v in video_paths]

    parser = argparse.ArgumentParser(description="Create grid GIF from multiple MP4 videos")
    parser.add_argument("--output", default="grid_comparison.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=16, help="FPS for output GIF")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process from each video")
    parser.add_argument("--ncol", type=int, default=4, help="Number of columns in the grid layout")
    
    args = parser.parse_args()
    
    # Check if all video files exist
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            print("Please update the hard-coded video_paths list in the script")
            return
    
    # Display layout information
    print(f"Creating grid with {len(video_paths)} videos in {args.ncol} columns")
    print("Layout order: Left to right, top to bottom")
    print("Row 0: Videos 0, 1, 2, 3, 4")
    print("Row 1: Videos 5, 6, 7, 8, 9")
    print("Row 2: Videos 10, 11, 12, 13, 14")
    print("Row 3: Videos 15, 16, 17, 18, 19")
    print("Row 4: Videos 20, 21, 22, 23, 24")
    print("Row 5: Video 25")
    print()
    
    create_grid_gif(
        video_paths=video_paths,
        labels=labels,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        ncol=args.ncol
    )

if __name__ == "__main__":
    main()
