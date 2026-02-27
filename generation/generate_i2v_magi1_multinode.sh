# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s


#!/bin/bash

# SLURM job array configuration for multi-node MAGI-1 execution
#SBATCH --job-name=magi1_phy
#SBATCH --array=0-0                    # 1 nodes (0)
#SBATCH --nodes=1                      # Each job uses 1 node
#SBATCH --qos=h200_dream_high
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --gres=gpu:8                  # 8 GPUs per node
#SBATCH --cpus-per-task=48            # Adjust based on your cluster
#SBATCH --mem=512G                    # Adjust based on your cluster
#SBATCH --time=24:00:00               # Adjust based on expected runtime
#SBATCH --output=jobs/job_%A_%a.out
#SBATCH --error=jobs/job_%A_%a.err

source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
conda activate vg

nvidia-smi

# Multi-node configuration
NUM_NODES=1                           # Total number of nodes
NUM_GPUS_PER_NODE=8                   # GPUs per node
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))  # 8 total GPUs
NODE_ID=${SLURM_ARRAY_TASK_ID}        # Current node ID (0-3)
# NODE_ID=0

echo "Starting node ${NODE_ID} of ${NUM_NODES} (GPUs per node: ${NUM_GPUS_PER_NODE}, Total GPUs: ${TOTAL_GPUS})"

# Define hyperparameter triplets (slice_window_size, vjepa_context_frames, slice_stride)
# For vanilla: these are harmless; for guidance: they control the sliding window
TRIPLETS=(
    "16 8 8"   # window=16, context_frames=8, stride=8
)

SEED_LIST=(42)

GUIDANCE_STEP_PATTERN="0x5,1x45"
GUIDANCE_LR_PATTERNS=("0.001x50")
GUIDANCE_FREQUENCY=1

# CFG scale values for classifier-free guidance ablation
CFG_SCALES=("6.0")

# Disable Time Travel for simple algorithm
GUIDANCE_RANGES=("0 0")

# Path to MAGI-1 config file
MAGI1_CONFIG_FILE="./MAGI-1/configs/inference_config.json"

# JSON batch describing entries with input image/video, prompt, and output path
# Add or remove batch JSON files as needed
BATCH_JSON_LIST=(
    # Physics-IQ dataset
    "./prompts/physics_iq.json"
)
BASEDIR="./physicsiq_benchmark/code"
OUTPUT_FOLDER="./generated_videos"

# SAMPLE_METHODS=("guidance" "vanilla")
SAMPLE_METHODS=("guidance")
NUM_SAMPLING_STEPS="50"
NUM_FRAMES="49"
REJECTION_SAMPLES="10"  # Number of candidates to generate for rejection sampling

# I2V conditioning comes from JSON (input_video or image); no static INIT_IMAGE here

# V-JEPA slice-pred fixed settings
VJEPA_VARIANTS=("vit_giant")
VJEPA_IMG_SIZE=256
VJEPA_MASKING_MODE="causal"
# Loss aggregation modes to iterate over
LOSS_MODES=("mean")


mkdir -p "$OUTPUT_FOLDER"

for BATCH_JSON in "${BATCH_JSON_LIST[@]}"; do
for SAMPLE_METHOD in "${SAMPLE_METHODS[@]}"; do

    for triplet in "${TRIPLETS[@]}"; do
            # Split triplet into individual variables
            read -r SLICE_WINDOW_SIZE CONTEXT_LENGTH STRIDE <<< "$triplet"
            for guidance_range in "${GUIDANCE_RANGES[@]}"; do
                # Split guidance range into start and end values (GLOBAL 0..49)
                read -r GUIDANCE_START GUIDANCE_END <<< "$guidance_range"
                TRAVEL_TIME="${GUIDANCE_START},${GUIDANCE_END}"
                for CFG_SCALE in "${CFG_SCALES[@]}"; do
                    for LOSS_MODE in "${LOSS_MODES[@]}"; do
                    for VJEPA_VARIANT in "${VJEPA_VARIANTS[@]}"; do
                    echo "Config: Method=$SAMPLE_METHOD, Context=$CONTEXT_LENGTH, Stride=$STRIDE, Range=$TRAVEL_TIME, CFG=$CFG_SCALE, LossMode=$LOSS_MODE, VJEPA=$VJEPA_VARIANT"

                    # Match structure: <OUTPUT_FOLDER>/<group>/<experiment>/<name>.mp4
                    if [[ "$(basename "$BATCH_JSON")" == "physics_iq.json" ]]; then
                        GROUP_NAME="physics_iq"
                    elif [[ "$(basename "$BATCH_JSON")" == "physics_iq_multiframe.json" ]]; then
                        GROUP_NAME="physics_iq_multiframe"
                    else
                        GROUP_NAME=$(basename "$(dirname "$BATCH_JSON")")
                    fi
                    MODEL_OUTPUT_FOLDER="${OUTPUT_FOLDER}/${GROUP_NAME}/MAGI-1"
                    mkdir -p "$MODEL_OUTPUT_FOLDER"

                    # Loop over LR patterns; pass base output folder and let Python name runs
                    for GUIDANCE_LR_PATTERN in "${GUIDANCE_LR_PATTERNS[@]}"; do
                        RUN_OUTPUT_FOLDER="$MODEL_OUTPUT_FOLDER"
                        mkdir -p "$RUN_OUTPUT_FOLDER"

                        # Launch one worker per GPU on this node; each worker shards the JSON by global index
                        for SEED in "${SEED_LIST[@]}"; do
                        for ((g=0; g<NUM_GPUS_PER_NODE; g++)); do
                            # Calculate global GPU index across all nodes
                            GLOBAL_GPU_IDX=$((NODE_ID * NUM_GPUS_PER_NODE + g))
                            echo "  -> Launching worker on Node $NODE_ID, Local GPU $g (Global GPU $GLOBAL_GPU_IDX) with LR pattern $GUIDANCE_LR_PATTERN"
                            CUDA_VISIBLE_DEVICES=$g python generator_i2v_multinode.py \
                                --config_file "$MAGI1_CONFIG_FILE" \
                                --output_folder "$RUN_OUTPUT_FOLDER" \
                                --batch_json "$BATCH_JSON" \
                                --base_dir "$BASEDIR" \
                                --num_gpus $TOTAL_GPUS \
                                --gpu_idx $GLOBAL_GPU_IDX \
                                --num_nodes $NUM_NODES \
                                --node_id $NODE_ID \
                                --gpus_per_node $NUM_GPUS_PER_NODE \
                                --sampling_method "$SAMPLE_METHOD" \
                                --num_inference_steps $NUM_SAMPLING_STEPS \
                                --num_frames 49 \
                                --height 480 \
                                --width 720 \
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
                                --loss_mode "$LOSS_MODE" \
                                --rejection_samples $REJECTION_SAMPLES \
                                --config_version "v2" \
                                --seed $SEED &
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
