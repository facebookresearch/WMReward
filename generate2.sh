
#!/bin/bash

# Define hyperparameter triplets (kernel_size, context_length, stride)
# For vanilla: parameters are ignored, so use dummy values
# For rejection/guidance: use meaningful V-JEPA parameters
TRIPLETS=(
    "8 6 2"   # Good balance for V-JEPA evaluation
    # "4 2 2"   # Shorter context
    # "16 10 4" # Longer context
)

MODEL_NAMES=("wan")
PROMPT_FILES=("subject_consistency")
NUM_GPUS=8
OUTPUT_FOLDER="generated_videos"
SAMPLE_METHODS=("vanilla" "rejection" "guidance")
NUM_SAMPLING_STEPS="50"

# Ablation arrays
CFG_SCALES=(3.0 5.0 7.5 10.0)                    # Classifier-free guidance scale
GUIDANCE_RHO_SCALES=(1.0 3.0 5.0 10.0)          # Rho scale for physics guidance

# Ablation parameters (adjust these for different experiments):
# --num_rejection_attempts: Number of attempts for rejection sampling
# --vjepa_mode: V-JEPA aggregation mode ('mean' or 'max')
# --cfg_scale: Classifier-free guidance scale (for all methods)
# --guidance_start/end: Timestep range for applying guidance
# --guidance_rho_scale: Gradient scaling factor (rho_scale) for guidance

mkdir -p "$OUTPUT_FOLDER"

for SAMPLE_METHOD in "${SAMPLE_METHODS[@]}"; do 
    for CFG_SCALE in "${CFG_SCALES[@]}"; do
        for GUIDANCE_RHO_SCALE in "${GUIDANCE_RHO_SCALES[@]}"; do
            # Skip non-default rho values for non-guidance methods (efficiency)
            if [[ "$SAMPLE_METHOD" != "guidance" && "$GUIDANCE_RHO_SCALE" != "3.0" ]]; then
                continue
            fi
            
            echo "🚀 Running ablation: Method=$SAMPLE_METHOD, CFG=$CFG_SCALE, Rho=$GUIDANCE_RHO_SCALE"
            
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
                                --num_frames 17 \
                                --num_rejection_attempts 10 \
                                --vjepa_mode "max" \
                                --cfg_scale $CFG_SCALE \
                                --guidance_start 0 \
                                --guidance_end 1001 \
                                --guidance_rho_scale $GUIDANCE_RHO_SCALE &
                        done
                        wait
                    done
                done
            done
        done
    done
done