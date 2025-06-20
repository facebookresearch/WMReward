import os
import csv

# Directory containing the video files
directory = '/home/yjianhao/project/video_guidance/generated_videos/wan2.1-1.3b/subject_consistency'

# Output CSV file
output_csv = 'sapc_wan.csv'

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

# Write the rows to the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['caption', 'videopath'])
    # Write the data
    csvwriter.writerows(rows)

print(f"CSV file '{output_csv}' has been created.")