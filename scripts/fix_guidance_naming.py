#!/usr/bin/env python3

import os
import re
from pathlib import Path

def fix_guidance_naming():
    """Fix naming inconsistency in guidance_v1_f49_s50_cfg6.0_trimmed folder."""
    
    # Define paths
    guidance_dir = Path("/home/yjianhao/project/frame-guidance/generated_videos/physics_iq/cogvideox5b_i2v/guidance_v1_f49_s50_cfg6.0_trimmed")
    vanilla_dir = Path("/home/yjianhao/project/frame-guidance/generated_videos/physics_iq/cogvideox5b_i2v/vanilla_f49_s50_cfg6.0_trimmed")
    
    if not guidance_dir.exists():
        print(f"Guidance directory not found: {guidance_dir}")
        return
    
    if not vanilla_dir.exists():
        print(f"Vanilla directory not found: {vanilla_dir}")
        return
    
    # Get all files from guidance folder
    guidance_files = sorted([f for f in guidance_dir.glob("*.mp4")])
    
    # Define perspective pattern based on vanilla folder
    perspectives = ["perspective-left", "perspective-center", "perspective-right"]
    
    print(f"Found {len(guidance_files)} files in guidance folder")
    print("Starting to rename files...")
    
    rename_count = 0
    
    for i, file_path in enumerate(guidance_files):
        # Extract the current filename pattern: NNNN_trimmed-scenario.mp4
        match = re.match(r'(\d{4})_trimmed-(.+)\.mp4', file_path.name)
        if not match:
            print(f"Warning: Unexpected filename format: {file_path.name}")
            continue
        
        file_number = match.group(1)
        scenario = match.group(2)
        
        # Determine perspective based on file position
        # Every 3 consecutive files represent the same scenario from different perspectives
        perspective_index = i % 3
        perspective = perspectives[perspective_index]
        
        # Create new filename: NNNN_perspective-{view}_trimmed-{scenario}.mp4
        new_filename = f"{file_number}_{perspective}_trimmed-{scenario}.mp4"
        new_path = guidance_dir / new_filename
        
        # Check if target already exists
        if new_path.exists():
            print(f"Warning: Target file already exists, skipping: {new_filename}")
            continue
        
        # Rename the file
        try:
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_filename}")
            rename_count += 1
        except Exception as e:
            print(f"Error renaming {file_path.name}: {e}")
    
    print(f"\nCompleted! Renamed {rename_count} files")
    
    # Verify the pattern matches vanilla folder
    print("\nVerifying pattern consistency...")
    guidance_files_new = sorted([f for f in guidance_dir.glob("*.mp4")])
    vanilla_files = sorted([f for f in vanilla_dir.glob("*.mp4")])
    
    # Check if patterns now match
    matching_patterns = 0
    for guidance_file, vanilla_file in zip(guidance_files_new[:10], vanilla_files[:10]):  # Sample check
        guidance_pattern = re.sub(r'^\d{4}_', 'NNNN_', guidance_file.name)
        vanilla_pattern = re.sub(r'^\d{4}_', 'NNNN_', vanilla_file.name)
        
        if guidance_pattern == vanilla_pattern:
            matching_patterns += 1
        else:
            print(f"Pattern mismatch: {guidance_pattern} vs {vanilla_pattern}")
    
    print(f"Pattern consistency check: {matching_patterns}/10 files match expected pattern")

if __name__ == "__main__":
    fix_guidance_naming()
