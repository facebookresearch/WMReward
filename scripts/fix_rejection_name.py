#!/usr/bin/env python3
"""
Post-hoc script to rename rejection sampling videos to match vanilla video indices.
This ensures consistent numbering for easy comparison and visualization.

Usage:
    python fix_rejection_name.py --vanilla_dir <vanilla_folder> --rejection_dir <rejection_folder>
    
Example:
    python fix_rejection_name.py \
        --vanilla_dir /path/to/vanilla_f49_s50_cfg6.0 \
        --rejection_dir /path/to/rejection_v1_f49_s50_cfg6.0_reject10_max
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def extract_prompt_from_filename(filename):
    """Extract the prompt part from a video filename, removing any index prefix."""
    # Remove .mp4 extension
    name = Path(filename).stem
    
    # Remove index prefix if present (e.g., "001_prompt" -> "prompt")
    # Match pattern: digits followed by underscore at the start
    match = re.match(r'^\d+_(.+)$', name)
    if match:
        return match.group(1)
    else:
        # No index prefix, return as is
        return name


def get_vanilla_mapping(vanilla_dir):
    """Create a mapping from prompt to index based on vanilla video filenames."""
    prompt_to_index = {}
    
    if not os.path.exists(vanilla_dir):
        raise ValueError(f"Vanilla directory not found: {vanilla_dir}")
    
    for filename in sorted(os.listdir(vanilla_dir)):
        if filename.endswith('.mp4'):
            prompt = extract_prompt_from_filename(filename)
            
            # Extract index from filename (e.g., "001_prompt.mp4" -> 1)
            name = Path(filename).stem
            match = re.match(r'^(\d+)_', name)
            if match:
                index = int(match.group(1))
                prompt_to_index[prompt] = index
                print(f"Vanilla mapping: '{prompt}' -> {index:03d}")
            else:
                print(f"Warning: No index found in vanilla filename: {filename}")
    
    return prompt_to_index


def rename_rejection_videos(rejection_dir, prompt_to_index, dry_run=False):
    """Rename rejection videos to match vanilla indices."""
    if not os.path.exists(rejection_dir):
        raise ValueError(f"Rejection directory not found: {rejection_dir}")
    
    renamed_count = 0
    skipped_count = 0
    
    for filename in sorted(os.listdir(rejection_dir)):
        if filename.endswith('.mp4'):
            prompt = extract_prompt_from_filename(filename)
            
            if prompt in prompt_to_index:
                target_index = prompt_to_index[prompt]
                new_filename = f"{target_index:03d}_{prompt}.mp4"
                
                old_path = os.path.join(rejection_dir, filename)
                new_path = os.path.join(rejection_dir, new_filename)
                
                if filename != new_filename:
                    print(f"Rename: {filename} -> {new_filename}")
                    if not dry_run:
                        if os.path.exists(new_path):
                            print(f"  Warning: Target file already exists: {new_filename}")
                            backup_path = new_path + ".backup"
                            shutil.move(new_path, backup_path)
                            print(f"  Backed up existing file to: {backup_path}")
                        shutil.move(old_path, new_path)
                    renamed_count += 1
                else:
                    print(f"Already correct: {filename}")
            else:
                print(f"Warning: No vanilla mapping found for prompt '{prompt}' in file: {filename}")
                skipped_count += 1
    
    print(f"\nSummary:")
    print(f"  Renamed: {renamed_count} files")
    print(f"  Skipped: {skipped_count} files")
    if dry_run:
        print(f"  (Dry run - no actual changes made)")


def main():
    parser = argparse.ArgumentParser(description="Rename rejection sampling videos to match vanilla indices")
    parser.add_argument('--vanilla_dir', type=str, 
                       default='/home/yjianhao/project/frame-guidance/generated_videos/gr1_behavior/cogvideox5b_i2v/vanilla_f49_s50_cfg6.0',
                       help='Path to vanilla video folder (contains indexed videos)')
    parser.add_argument('--rejection_dir', type=str, 
                       default='/home/yjianhao/project/frame-guidance/generated_videos/gr1_behavior/cogvideox5b_i2v/rejection_v1_f49_s50_cfg6.0_reject10_max',
                       help='Path to rejection video folder (to be renamed)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be renamed without making changes')
    
    args = parser.parse_args()
    
    print(f"Vanilla directory: {args.vanilla_dir}")
    print(f"Rejection directory: {args.rejection_dir}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Step 1: Extract vanilla prompt-to-index mapping
    print("=== Step 1: Analyzing vanilla videos ===")
    prompt_to_index = get_vanilla_mapping(args.vanilla_dir)
    print(f"Found {len(prompt_to_index)} vanilla videos with indices")
    print()
    
    # Step 2: Rename rejection videos
    print("=== Step 2: Renaming rejection videos ===")
    rename_rejection_videos(args.rejection_dir, prompt_to_index, dry_run=args.dry_run)
    print()
    
    if args.dry_run:
        print("Re-run without --dry_run to actually perform the renaming.")
    else:
        print("Renaming complete!")


if __name__ == "__main__":
    main()
