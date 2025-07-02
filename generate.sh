#!/bin/bash

# Define hyperparameter triplets (kernel_size, context_length, stride)
# For vanilla: parameters are ignored, so use dummy values
# For rejection/guidance: use meaningful V-JEPA parameters
TRIPLETS=(
    "8 6 2"   # Good balance for V-JEPA evaluation
    # "4 2 2"   # Shorter context
    # "16 10 4" # Longer context
)

# Rho scale values for guidance ablation
RHO_SCALES=(
    "1.0"
    "3.0"
    "50.0"
    "100.0" 
    # "5.0"
    # "10.0"
    # "15.0"
    # "20.0"
)

MODEL_NAMES=("wan")
PROMPT_FILES=("subject_consistency")
NUM_GPUS=8
OUTPUT_FOLDER="generated_videos"
SAMPLE_METHODS=("vanilla" "rejection")
# SAMPLE_METHODS=("guidance")
NUM_SAMPLING_STEPS="50" 

# Ablation parameters (adjust these for different experiments):
# --num_rejection_attempts: Number of attempts for rejection sampling
# --vjepa_mode: V-JEPA aggregation mode ('mean' or 'max')
# --cfg_scale: Classifier-free guidance scale (for all methods)
# --guidance_start/end: Timestep range for applying guidance
# --guidance_rho_scale: Gradient scaling factor (rho_scale) for guidance

mkdir -p "$OUTPUT_FOLDER"

for SAMPLE_METHOD in "${SAMPLE_METHODS[@]}"; do 
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
        for triplet in "${TRIPLETS[@]}"; do
            # Split triplet into individual variables
            read -r KERNEL_SIZE CONTEXT_LENGTH STRIDE <<< "$triplet"
            
            for RHO_SCALE in "${RHO_SCALES[@]}"; do
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
                        --num_frames 33 \
                        --height 480 \
                        --width 832 \
                        --num_rejection_attempts 10 \
                        --vjepa_mode "max" \
                        --cfg_scale 5.0 \
                        --guidance_start 0 \
                        --guidance_end 1001 \
                        --guidance_rho_scale $RHO_SCALE &
                done
                wait
            done
        done
    done
done
done