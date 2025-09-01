#!/bin/bash

# SLURM job array configuration for multi-node execution
#SBATCH --job-name=c14bdrm_guide_multinode
#SBATCH --array=0-2                    # 4 nodes (0,1,2,3)
#SBATCH --nodes=1                      # Each job uses 1 node
#SBATCH --qos=dream_high
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --gres=gpu:8                  # 8 GPUs per node
#SBATCH --cpus-per-task=48            # Adjust based on your cluster
#SBATCH --mem=512G                    # Adjust based on your cluster
#SBATCH --time=24:00:00               # Adjust based on expected runtime
#SBATCH --output=jobs/cosmos_guide_node_%A_%a.out
#SBATCH --error=jobs/cosmos_guide_node_%A_%a.err

source /home/yjianhao/miniconda3/bin/activate
conda activate vg
nvidia-smi

# Multi-node configuration
NUM_NODES=3                           # Total number of nodes
NUM_GPUS_PER_NODE=8                   # GPUs per node
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))  # 32 total GPUs
NODE_ID=${SLURM_ARRAY_TASK_ID}        # Current node ID (0-3)

echo "Starting node ${NODE_ID} of ${NUM_NODES} (GPUs per node: ${NUM_GPUS_PER_NODE}, Total GPUs: ${TOTAL_GPUS})"

# Define hyperparameter triplets (slice_window_size, vjepa_context_frames, slice_stride)
# For vanilla: these are harmless; for guidance: they control the sliding window
TRIPLETS=(
    "16 8 8"   # window=16, context_frames=8, stride=4
)

# Guidance step/lr patterns and frequency (match run_vjepa_slicepred.py defaults)
# GUIDANCE_STEP_PATTERN="0x3,2x12,1x10,1x10"
# GUIDANCE_LR_PATTERN="4.0x15,3.0x10,2.0x10"
# GUIDANCE_STEP_PATTERN="0x35"
# GUIDANCE_LR_PATTERN="0x35"
GUIDANCE_STEP_PATTERN="0x3,2x12,3x10,1x10"
GUIDANCE_LR_PATTERN="0.001x35"
GUIDANCE_FREQUENCY=1

# CFG scale values for classifier-free guidance ablation
CFG_SCALES=(
    # "1.0"
    # "2.0"
    # "3.0"
    # "6.0"
    "7.0"
    # "10.0"
)

# Guidance range values for range ablation (format: "start end") in GLOBAL steps [0..49]
GUIDANCE_RANGES=(
    "0 0"       
)

# MODEL_NAMES=("nvidia/Cosmos-Predict2-2B-Video2World")
MODEL_NAMES=("nvidia/Cosmos-Predict2-14B-Video2World")

# JSON batch describing entries with input image/video, prompt, and output path
# Add or remove batch JSON files as needed
BATCH_JSON_LIST=(
    # DreamBench datasets
    "/home/yjianhao/project/cosmos-predict2/dream_gen_benchmark/gr1_object/batch_input.json"
    "/home/yjianhao/project/cosmos-predict2/dream_gen_benchmark/gr1_env/batch_input.json"
    "/home/yjianhao/project/cosmos-predict2/dream_gen_benchmark/gr1_behavior/batch_input.json"
    
    # Physics-IQ dataset 
    # "/home/yjianhao/project/frame-guidance/prompts/physics_iq.json"
    # "/home/yjianhao/project/frame-guidance/prompts/physics_iq_5frame.json"
)
BASEDIR="/home/yjianhao/project/cosmos-predict2"
OUTPUT_FOLDER="/home/yjianhao/project/frame-guidance/generated_videos"
SAMPLE_METHODS=("guidance")
# SAMPLE_METHODS=("rejection")
# SAMPLE_METHODS=("guidance")
NUM_SAMPLING_STEPS="35"
REJECTION_SAMPLES="10"  # Number of candidates to generate for rejection sampling 

# I2V conditioning comes from JSON (input_video or image); no static INIT_IMAGE here

# V-JEPA slice-pred fixed settings
VJEPA_VARIANT="vit_giant"
VJEPA_IMG_SIZE=256
VJEPA_MASKING_MODE="causal"
VJEPA_MASK_RATIO=0.75
STYLE_WEIGHT=1.0

# Ablation parameters (adjust these for different experiments):
# --rejection_samples: Number of candidates to generate for rejection sampling
# --vjepa_mode: V-JEPA aggregation mode ('mean' or 'max')
# --cfg_scale: Classifier-free guidance scale (for all methods)
# --guidance_start/end: Timestep range for applying guidance (now looped via GUIDANCE_RANGES)
# --guidance_rho_scale: Gradient scaling factor (rho_scale) for guidance (looped via RHO_SCALES)
# --vjepa_model_type: Use 'torch' for Quentin's V-JEPA implementation
#
# This script now performs a full ablation study over:
# - RHO_SCALES: Different guidance strength values
# - GUIDANCE_RANGES: Different timestep ranges for when to apply guidance

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
                        echo "Config: Method=$SAMPLE_METHOD, Model=$MODEL_BASE_NAME, Context=$CONTEXT_LENGTH, Stride=$STRIDE, Range=$TRAVEL_TIME, CFG=$CFG_SCALE"

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

                        # Launch one worker per GPU on this node; each worker shards the JSON by global index
                        for ((g=0; g<NUM_GPUS_PER_NODE; g++)); do
                            # Calculate global GPU index across all nodes
                            GLOBAL_GPU_IDX=$((NODE_ID * NUM_GPUS_PER_NODE + g))
                            echo "  -> Launching worker on Node $NODE_ID, Local GPU $g (Global GPU $GLOBAL_GPU_IDX)"
                            CUDA_VISIBLE_DEVICES=$g python generator_i2v_multinode.py \
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
                                --vjepa_mask_ratio $VJEPA_MASK_RATIO \
                                --style_weight $STYLE_WEIGHT \
                                --vjepa_context_frames $CONTEXT_LENGTH \
                                --slice_stride $STRIDE \
                                --slice_window_size $SLICE_WINDOW_SIZE \
                                --guidance_step_pattern "$GUIDANCE_STEP_PATTERN" \
                                --guidance_lr_pattern "$GUIDANCE_LR_PATTERN" \
                                --guidance_frequency $GUIDANCE_FREQUENCY \
                                --loss_mode "max" \
                                --rejection_samples $REJECTION_SAMPLES \
                                --travel_time "$TRAVEL_TIME" &
                        done
                        wait
                    done
                done
        done
        done
    done
done
done

echo "Node ${NODE_ID} experiments completed! Results saved to: $OUTPUT_FOLDER"
