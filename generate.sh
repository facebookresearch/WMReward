
#!/bin/bash

# Define hyperparameter triplets (kernel_size, context_length, stride)
TRIPLETS=(
    # "4 2 2"   # Triplet 1
    # "4 3 1"   # Triplet 4
    # "8 4 4"   # Triplet 2
    # "8 6 2"   # Triplet 3
    # "16 10 6"
    "8 4 2"
)

MODEL_NAMES=("wan")
PROMPT_FILES=("subject_consistency")
NUM_GPUS=8
OUTPUT_FOLDER="generated_videos"
SAMPLE_METHOD="rejection"
# SAMPLE_METHOD="vanilla"
NUM_SAMPLING_STEPS="50"

mkdir -p "$OUTPUT_FOLDER"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
        for triplet in "${TRIPLETS[@]}"; do
            # Split triplet into individual variables
            read -r KERNEL_SIZE CONTEXT_LENGTH STRIDE <<< "$triplet"
            
            # Create unique output folder
            MODEL_OUTPUT_FOLDER="${OUTPUT_FOLDER}"
            mkdir -p "$MODEL_OUTPUT_FOLDER"

            # Run on GPUs
            for ((i=0; i<NUM_GPUS; i++)); do
                CUDA_VISIBLE_DEVICES=$i python generator.py \
                    "$PROMPT_FILE" \
                    "$MODEL_NAME" \
                    "$MODEL_OUTPUT_FOLDER" \
                    "$NUM_GPUS" \
                    "$i" \
                    "$SAMPLE_METHOD" \
                    "$KERNEL_SIZE" \
                    "$CONTEXT_LENGTH" \
                    "$STRIDE" \
                    "$NUM_SAMPLING_STEPS" &
            done
            wait
        done
    done
done