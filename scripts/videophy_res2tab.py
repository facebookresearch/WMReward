import pandas as pd
import glob
import os
import argparse

args = argparse.ArgumentParser(description="Calculate average scores from CSV files.")
args.add_argument('--models', type=str, nargs='+', default=['wan2.1-1.3b', 'wan2.1-1.3b-rej'], help='List of model names to process.')

# Directory containing the CSV files
for model in models:
    directory_path = f'/home/yjianhao/project/video_guidance/results/videophy/{model}'

    # Pattern to match CSV files in the directory
    csv_files_pattern = os.path.join(directory_path, '*.csv')

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(csv_files_pattern)

    # Iterate over each CSV file and calculate the average score
    for csv_file in csv_files:
        # Read the CSV data into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Calculate the average score
        average_score = df['score'].mean()
        
        # Print the average score for each file
        print(f"The average score in '{os.path.basename(csv_file)}' is: {average_score}")