#!/usr/bin/env python3
"""
Script to sort and deduplicate captions from a CSV file.
Reads captions from prompt-upsampled-test.csv and outputs sorted, deduplicated captions to a txt file.
"""

import csv
from pathlib import Path

def process_captions(csv_file_path, output_file_path, column_choice='both', sort_by_length=False):
    """
    Process captions from CSV file: extract, deduplicate, sort, and save to txt file.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str): Path to the output txt file
        column_choice (str): Which column to process ('caption', 'upsampled_caption', or 'both')
        sort_by_length (bool): If True, sort by caption length; if False, sort alphabetically
    """
    captions = set()  # Using set for automatic deduplication
    
    # Read CSV file
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                if column_choice == 'caption' or column_choice == 'both':
                    if row['caption'].strip():  # Only add non-empty captions
                        captions.add(row['caption'].strip().strip('"'))
                
                if column_choice == 'upsampled_caption' or column_choice == 'both':
                    if row['upsampled_caption'].strip():  # Only add non-empty captions
                        captions.add(row['upsampled_caption'].strip().strip('"'))
        
        print(f"Extracted {len(captions)} unique captions from {csv_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found!")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Convert set to list and sort
    captions_list = list(captions)
    
    if sort_by_length:
        captions_list.sort(key=len)
        print("Sorted captions by length")
    else:
        captions_list.sort()
        print("Sorted captions alphabetically")
    
    # Write to output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for caption in captions_list:
                file.write(caption + '\n')
        
        print(f"Successfully saved {len(captions_list)} unique captions to {output_file_path}")
        
    except Exception as e:
        print(f"Error writing to output file: {e}")

def main():
    # Hardcoded paths
    # input_file = "videophy_test_public.csv"
    input_file = "videophy2_test.csv"
    output_file = "./prompts/videophy2_test.txt"
    
    print(f"Processing captions from: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print("Processing caption column only")
    print("Sorting alphabetically")
    print("-" * 50)
    
    # Process captions with default settings
    process_captions(input_file, output_file, column_choice='caption', sort_by_length=False)

if __name__ == "__main__":
    main() 