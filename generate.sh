
#!/bin/bash

# Define hyperparameter triplets (kernel_size, context_length, stride)
TRIPLETS=(
    # "4 2 2"   # Triplet 1
    # "4 3 1"   # Triplet 4
    # "8 4 4"   # Triplet 2
    # "8 6 2"   # Triplet 3
    # "16 10 6"
    # "8 4 2"
    "0 0 0"
)

MODEL_NAMES=("wan")
PROMPT_FILES=("subject_consistency")
NUM_GPUS=8
OUTPUT_FOLDER="generated_videos"
# SAMPLE_METHOD="rejection"
SAMPLE_METHODS=("rejection")
# SAMPLE_METHOD="vanilla"
NUM_SAMPLING_STEPS="50"

mkdir -p "$OUTPUT_FOLDER"

for SAMPLE_METHOD in "${SAMPLE_METHODS[@]}"; do 
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
                CUDA_VISIBLE_DEVICES=$i python generator.py --prompt_file $PROMPT_FILE \
                    --model_id $MODEL_NAME \
                    --output_folder $MODEL_OUTPUT_FOLDER \
                    --num_gpus $NUM_GPUS \
                    --gpu_idx $i \
                    --sampling_method $SAMPLE_METHOD \
                    --kernel_size $KERNEL_SIZE \
                    --context_length $CONTEXT_LENGTH \
                    --stride $STRIDE \
                    --num_inference_steps $NUM_SAMPLING_STEPS \
                    --num_frame 17 &
            done
            wait
        done
    done
    done
done