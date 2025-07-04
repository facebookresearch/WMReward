import cv2
import sys
import os
import numpy as np
from pathlib import Path

def crop_black_edges(frame, threshold=10):
    """
    Crop black edges from the frame.
    
    Args:
        frame: Input frame (numpy array)
        threshold: Pixel value threshold to consider as "black" (default: 10)
    
    Returns:
        Cropped frame
    """
    # Convert to grayscale for easier edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find all non-black pixels
    non_black_pixels = np.where(gray > threshold)
    
    if len(non_black_pixels[0]) == 0:
        # If all pixels are black, return original frame
        return frame
    
    # Find bounding box of non-black pixels
    top = np.min(non_black_pixels[0])
    bottom = np.max(non_black_pixels[0])
    left = np.min(non_black_pixels[1])
    right = np.max(non_black_pixels[1])
    
    # Crop the frame
    cropped_frame = frame[top:bottom+1, left:right+1]
    
    print(f"Cropped from {frame.shape[:2]} to {cropped_frame.shape[:2]} (removed black edges)")
    
    return cropped_frame

def save_first_frame(video_path, output_path=None):
    """
    Load an MP4 video and save the 24th frame as a PNG image.
    
    Args:
        video_path (str): Path to the input MP4 video file
        output_path (str, optional): Path for the output PNG file. 
                                   If None, saves as '{video_name}_24th_frame.png'
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return False
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return False
    
    # Jump to the 24th frame (frame index 23, since it's 0-indexed)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 23)
    
    # Read the 24th frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read the 24th frame from the video.")
        cap.release()
        return False
    
    # Crop black edges from the frame
    cropped_frame = frame
    
    # Generate output path if not provided
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"{video_name}_24th_frame_base.png"
    
    # Save the cropped 24th frame as PNG
    success = cv2.imwrite(output_path, cropped_frame)
    
    # Release the video capture object
    cap.release()
    
    if success:
        print(f"Successfully saved 24th frame to: {output_path}")
        return True
    else:
        print(f"Error: Could not save frame to '{output_path}'.")
        return False

def main():
    """Main function with hardcoded video path."""
    # Hardcoded video path
    video_path = "/home/yjianhao/project/video_guidance/store/clean_droid_ood.mp4"
    
    print(f"Processing video: {video_path}")
    success = save_first_frame(video_path)
    
    if success:
        print("Done!")
    else:
        print("Failed to extract first frame.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
