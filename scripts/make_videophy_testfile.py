import os
import csv
import argparse

args = argparse.ArgumentParser(description="Generate a CSV file for video paths and captions.")
args.add_argument('--model_cat', type=str, default='cosmos_i2v', help='Model category.')
args.add_argument('--model', type=str, default='wan2.1-1.3b', help='Model name to use for video generation.') 
args.add_argument('--prompt', type=str, default='subject_consistency', help='Prompt type/category.')
args = args.parse_args()

# Directory containing the video files - corrected path structure
directory = f'/home/yjianhao/project/frame-guidance/generated_videos/{args.prompt}/{args.model_cat}/{args.model}'

# Output CSV file
output_csv = f'/home/yjianhao/project/frame-guidance/temp/{args.prompt}/{args.model}.csv'

# Create temp directory if it doesn't exist
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Check if directory exists
if not os.path.exists(directory):
    print(f"Error: Directory not found: {directory}")
    print("Please check if videos have been generated for this model.")
    exit(1)

# List to store the rows for the CSV
rows = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.mp4'):
        # Extract the caption from the filename
        caption = filename.replace('.mp4', '').replace('_', ' ')
        
        # Construct the full path to the video file
        videopath = os.path.join(directory, filename)
        
        # Append the row to the list
        rows.append([caption, videopath])

if not rows:
    print(f"Warning: No .mp4 files found in {directory}")
    exit(1)

# Write the rows to the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['caption', 'videopath'])
    # Write the data
    csvwriter.writerows(rows)

print(f"CSV file '{output_csv}' has been created with {len(rows)} video entries.")