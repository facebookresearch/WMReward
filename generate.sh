#!/bin/bash

# Define lists of model names and prompt files
MODEL_NAMES=("wan")
PROMPT_FILES=("subject_consistency")
NUM_GPUS=8
OUTPUT_FOLDER="generated_videos"
SAMPLE_METHOD="rejection"
CONTEXT_LENGTHS=(2 4 8 16)
# Create the output folder if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Loop over each model name and prompt file
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
    for CONTEXT_LENGTH in "${CONTEXT_LENGTHS[@]}"; do
        # Create a subfolder for each model and prompt combination
        MODEL_OUTPUT_FOLDER="${OUTPUT_FOLDER}"
        mkdir -p $MODEL_OUTPUT_FOLDER

        # Loop over each GPU index and run the Python script
        for ((i=0; i<NUM_GPUS; i++)); do
            # Set the CUDA_VISIBLE_DEVICES environment variable
            CUDA_VISIBLE_DEVICES=$i python generator.py $PROMPT_FILE $MODEL_NAME $MODEL_OUTPUT_FOLDER $NUM_GPUS $i $SAMPLE_METHOD $CONTEXT_LENGTH &
        done

        # Wait for all background processes to finish
        wait
    done
    done
done