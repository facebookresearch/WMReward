#!/bin/bash

# SLURM job array configuration for multi-node execution
#SBATCH --job-name=rej2b_pi_multi
#SBATCH --array=0-3                    # 1 nodes (0)
#SBATCH --nodes=1                      # Each job uses 1 node
#SBATCH --qos=dream_high
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --gres=gpu:8                  # 8 GPUs per node
#SBATCH --cpus-per-task=48            # Adjust based on your cluster
#SBATCH --mem=512G                    # Adjust based on your cluster
#SBATCH --time=24:00:00               # Adjust based on expected runtime
#SBATCH --output=jobs/rej_guide_node_%A_%a.out
#SBATCH --error=jobs/rej_guide_node_%A_%a.err

source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
conda activate vg
nvidia-smi

# Multi-node configuration
NUM_NODES=4                       # Total number of nodes
NUM_GPUS_PER_NODE=8                   # GPUs per node
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))  # 64 total GPUs
NODE_ID=${SLURM_ARRAY_TASK_ID}        # Current node ID (0-7)
# NODE_ID=0        # Current node ID (0-7)

echo "Starting node ${NODE_ID} of ${NUM_NODES} (GPUs per node: ${NUM_GPUS_PER_NODE}, Total GPUs: ${TOTAL_GPUS})"

# Rejection Sampling Script for I2V Generation
# This script runs rejection sampling experiments using V-JEPA loss evaluation
# Define hyperparameter triplets (slice_window_size, vjepa_context_frames, slice_stride)
# For rejection sampling: these control the V-JEPA loss computation window
TRIPLETS=(
    "16 8 8"   # window=16, context_frames=8, stride=4
)

# Guidance step/lr patterns and frequency (match run_vjepa_slicepred.py defaults)
GUIDANCE_STEP_PATTERN="0x3,1x32"
GUIDANCE_LR_PATTERN="0.002x35"
GUIDANCE_FREQUENCY=1

# CFG scale values for classifier-free guidance ablation
CFG_SCALES=(
    "7.0"
)

# Guidance range values for range ablation (format: "start end") in GLOBAL steps [0..49]
GUIDANCE_RANGES=(
    "0 0"       
)


MODEL_NAMES=("nvidia/Cosmos-Predict2-14B-Video2World")


# JSON batch describing entries with input image/video, prompt, and output path
# Add or remove batch JSON files as needed
BATCH_JSON_LIST=(

    # Physics-IQ dataset 
    "./prompts/physics_iq_felix.json"
)
BASEDIR="/checkpoint/dream/yjianhao/PhysicsIQ/code"
OUTPUT_FOLDER="/checkpoint/dream/yjianhao/generated_videos"
SAMPLE_METHODS=("rejection")
# SAMPLE_METHODS=("guidance" "vanilla" "rejection" "rej_guide")
NUM_SAMPLING_STEPS="50"

# Rejection sampling parameters (used when SAMPLE_METHODS includes "rejection" or "rej_guide")
# The script will automatically reuse existing candidates from higher-count experiments
REJECTION_SAMPLES=(
    16
) 

# I2V conditioning comes from JSON (input_video or image); no static INIT_IMAGE here

# V-JEPA settings for rejection sampling
VJEPA_VARIANT="vit_huge"
VJEPA_IMG_SIZE=256
VJEPA_MASKING_MODE="causal"
LOSS_MODES=("max")  # Loss aggregation modes for rejection sampling

# Ablation parameters for rejection sampling experiments:
# --rejection_samples: Number of candidate videos to generate for rejection sampling
# --cfg_scale: Classifier-free guidance scale for video generation
# --loss_mode: V-JEPA loss aggregation mode ('mean' or 'max')
# --vjepa_variant: V-JEPA model variant for loss computation
#
# AUTOMATIC CANDIDATE REUSE:
# The script automatically detects existing experiments with higher candidate counts
# and reuses candidates instead of regenerating. For example:
# - If you have 32 candidates and request 8, it will select best 8 from existing 32
# - If you have 32 candidates and request 50, it will generate 18 new candidates
# - This saves significant computation time for smaller rejection sample counts
#
# This script performs a full ablation study over:
# - REJECTION_SAMPLES: Different numbers of rejection samples (2, 4, 8, 16, 32, etc.)
# - CFG_SCALES: Different classifier-free guidance scales
# - VJEPA parameters: Context frames, stride, window size, loss mode

mkdir -p "$OUTPUT_FOLDER"

for BATCH_JSON in "${BATCH_JSON_LIST[@]}"; do
for SAMPLE_METHOD in "${SAMPLE_METHODS[@]}"; do 
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        # Extract model name for folder organization
        MODEL_BASE_NAME=$(basename "$MODEL_NAME")
        if [[ "$MODEL_BASE_NAME" == "CogVideoX-2b" ]]; then
            MODEL_BASE_NAME="cogvideox2b"
        fi
        if [[ "$MODEL_BASE_NAME" == "CogVideoX-5b-I2V" ]]; then
            MODEL_BASE_NAME="cogvideox5b_i2v"
        fi
        
        for triplet in "${TRIPLETS[@]}"; do
                # Split triplet into individual variables
                read -r SLICE_WINDOW_SIZE CONTEXT_LENGTH STRIDE <<< "$triplet"
                for guidance_range in "${GUIDANCE_RANGES[@]}"; do
                    # Split guidance range into start and end values (GLOBAL 0..49)
                    read -r GUIDANCE_START GUIDANCE_END <<< "$guidance_range"
                    TRAVEL_TIME="${GUIDANCE_START},${GUIDANCE_END}"
                    for CFG_SCALE in "${CFG_SCALES[@]}"; do
                        for LOSS_MODE in "${LOSS_MODES[@]}"; do
                        for REJECTION_SAMPLE in "${REJECTION_SAMPLES[@]}"; do
                        echo "Config: Method=$SAMPLE_METHOD, Model=$MODEL_BASE_NAME, Context=$CONTEXT_LENGTH, Stride=$STRIDE, Range=$TRAVEL_TIME, CFG=$CFG_SCALE, LossMode=$LOSS_MODE, RejectionSamples=$REJECTION_SAMPLE"

                        # Match structure: <OUTPUT_FOLDER>/<group>/<model>/<experiment>/<name>.mp4
                        # DreamGen JSONs live under .../dream_gen_benchmark/<group>/batch_input.json
                        # Physics-IQ JSON uses filename physics_iq.json; map to group 'physics_iq'
                        if [[ "$(basename "$BATCH_JSON")" == "physics_iq.json" ]]; then
                            GROUP_NAME="physics_iq"
                        else
                            GROUP_NAME=$(basename "$(dirname "$BATCH_JSON")")
                        fi
                        MODEL_OUTPUT_FOLDER="${OUTPUT_FOLDER}/${GROUP_NAME}/${MODEL_BASE_NAME}"
                        mkdir -p "$MODEL_OUTPUT_FOLDER"

                        # Build rejection sampling arguments
                        REJECTION_ARGS=""
                        if [ "$SAMPLE_METHOD" = "rejection" ] || [ "$SAMPLE_METHOD" = "rej_guide" ]; then
                            REJECTION_ARGS="--rejection_samples $REJECTION_SAMPLE"
                        fi

                        # Launch one worker per GPU on this node; each worker shards the JSON by global index
                        for ((g=0; g<NUM_GPUS_PER_NODE; g++)); do
                            # Calculate global GPU index across all nodes
                            GLOBAL_GPU_IDX=$((NODE_ID * NUM_GPUS_PER_NODE + g))
                            echo "  -> Launching worker on Node $NODE_ID, Local GPU $g (Global GPU $GLOBAL_GPU_IDX)"
                            CUDA_VISIBLE_DEVICES=$g python generator_i2v_rejection.py \
                                --model_id "$MODEL_NAME" \
                                --output_folder "$MODEL_OUTPUT_FOLDER" \
                                --batch_json "$BATCH_JSON" \
                                --base_dir "$BASEDIR" \
                                --num_gpus $TOTAL_GPUS \
                                --gpu_idx $GLOBAL_GPU_IDX \
                                --num_nodes $NUM_NODES \
                                --node_id $NODE_ID \
                                --gpus_per_node $NUM_GPUS_PER_NODE \
                                --sampling_method "$SAMPLE_METHOD" \
                                --num_inference_steps $NUM_SAMPLING_STEPS \
                                --num_frames 93 \
                                --height 704 \
                                --width 1280 \
                                --cfg_scale $CFG_SCALE \
                                --vjepa_variant $VJEPA_VARIANT \
                                --vjepa_img_size $VJEPA_IMG_SIZE \
                                --vjepa_masking_mode $VJEPA_MASKING_MODE \
                                --vjepa_context_frames $CONTEXT_LENGTH \
                                --slice_stride $STRIDE \
                                --slice_window_size $SLICE_WINDOW_SIZE \
                                --guidance_step_pattern "$GUIDANCE_STEP_PATTERN" \
                                --guidance_lr_pattern "$GUIDANCE_LR_PATTERN" \
                                --guidance_frequency $GUIDANCE_FREQUENCY \
                                --travel_time "$TRAVEL_TIME" \
                                --loss_mode "$LOSS_MODE" \
                                $REJECTION_ARGS &
                        done
                        wait
                        done
                        done
                    done
                done
        done
        done
    done
done
done

echo "Node ${NODE_ID} experiments completed! Results saved to: $OUTPUT_FOLDER"
