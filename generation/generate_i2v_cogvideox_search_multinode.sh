#!/bin/bash

# SLURM job array configuration for multi-node execution (Search methods: DSearch & SVDD)
#SBATCH --job-name=cogvideox_search
#SBATCH --array=0-3                    # nodes 0..3
#SBATCH --nodes=1                      # Each job uses 1 node
#SBATCH --qos=dream_high
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --gres=gpu:8                  # 8 GPUs per node
#SBATCH --cpus-per-task=48            # Adjust based on your cluster
#SBATCH --mem=0G                      # Adjust based on your cluster
#SBATCH --time=48:00:00               # Adjust based on expected runtime
#SBATCH --output=jobs/search_node_%A_%a.out
#SBATCH --error=jobs/search_%A_%a.err

source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
conda activate vg
nvidia-smi

# Multi-node configuration
NUM_NODES=4                      # Total number of nodes
NUM_GPUS_PER_NODE=8                   # GPUs per node
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
NODE_ID=${SLURM_ARRAY_TASK_ID}
# NODE_ID=0


echo "Starting node ${NODE_ID} of ${NUM_NODES} (GPUs per node: ${NUM_GPUS_PER_NODE}, Total GPUs: ${TOTAL_GPUS})"

# Search method configurations
SEARCH_METHODS=("svdd")

# "dsearch" 

# DSearch parameters
DSEARCH_NUM_BEAMS_LIST=(4)
DSEARCH_BRANCH_K_LIST=(2)
DSEARCH_STRIDE_LIST=(5)

# SVDD parameters  
SVDD_BRANCH_K_LIST=(2 4 8 16)
SVDD_BETA_LIST=(3.0)
SVDD_STRIDE_LIST=(5)

# Base model and data
MODEL_NAMES=("THUDM/CogVideoX-5b-I2V")
BATCH_JSON_LIST=(
    "./prompts/physics_iq.json"
)
BASEDIR="/checkpoint/dream/yjianhao/PhysicsIQ/code"
OUTPUT_FOLDER="/checkpoint/dream/yjianhao/generated_videos"

# Run config
NUM_SAMPLING_STEPS="50"
NUM_FRAMES="49"
CFG_SCALES=("6.0")
VJEPA_VARIANTS=("vit_giant")
VJEPA_IMG_SIZE=256
VJEPA_MASKING_MODE="causal"
LOSS_MODES=("mean")

mkdir -p "$OUTPUT_FOLDER"

for BATCH_JSON in "${BATCH_JSON_LIST[@]}"; do
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_BASE_NAME=$(basename "$MODEL_NAME")
    for CFG_SCALE in "${CFG_SCALES[@]}"; do
        for LOSS_MODE in "${LOSS_MODES[@]}"; do
        for VJEPA_VARIANT in "${VJEPA_VARIANTS[@]}"; do
            if [[ "$(basename "$BATCH_JSON")" == "physics_iq.json" ]]; then
                GROUP_NAME="physics_iq"
            elif [[ "$(basename "$BATCH_JSON")" == "physics_iq_multiframe.json" ]]; then
                GROUP_NAME="physics_iq_multiframe"
            else
                GROUP_NAME=$(basename "$(dirname "$BATCH_JSON")")
            fi
            MODEL_OUTPUT_FOLDER="${OUTPUT_FOLDER}/${GROUP_NAME}/${MODEL_BASE_NAME}"
            mkdir -p "$MODEL_OUTPUT_FOLDER"

            RUN_OUTPUT_FOLDER="$MODEL_OUTPUT_FOLDER"
            mkdir -p "$RUN_OUTPUT_FOLDER"

            # Loop over search methods
            for METHOD in "${SEARCH_METHODS[@]}"; do
                if [[ "$METHOD" == "dsearch" ]]; then
                    # DSearch parameter combinations
                    for NUM_BEAMS in "${DSEARCH_NUM_BEAMS_LIST[@]}"; do
                    for BRANCH_K in "${DSEARCH_BRANCH_K_LIST[@]}"; do
                    for STRIDE in "${DSEARCH_STRIDE_LIST[@]}"; do
                        # Launch one worker per GPU on this node
                        for ((g=0; g<NUM_GPUS_PER_NODE; g++)); do
                            GLOBAL_GPU_IDX=$((NODE_ID * NUM_GPUS_PER_NODE + g))
                            echo "  -> Launching DSearch worker on Node $NODE_ID, Local GPU $g (Global GPU $GLOBAL_GPU_IDX), B=$NUM_BEAMS, K=$BRANCH_K, stride=$STRIDE"
                            CUDA_VISIBLE_DEVICES=$g python generator_i2v_multinode.py \
                                --model_id "$MODEL_NAME" \
                                --output_folder "$RUN_OUTPUT_FOLDER" \
                                --batch_json "$BATCH_JSON" \
                                --base_dir "$BASEDIR" \
                                --num_gpus $TOTAL_GPUS \
                                --gpu_idx $GLOBAL_GPU_IDX \
                                --num_nodes $NUM_NODES \
                                --node_id $NODE_ID \
                                --gpus_per_node $NUM_GPUS_PER_NODE \
                                --sampling_method dsearch \
                                --num_inference_steps $NUM_SAMPLING_STEPS \
                                --num_frames $NUM_FRAMES \
                                --height 480 \
                                --width 720 \
                                --cfg_scale $CFG_SCALE \
                                --vjepa_variant $VJEPA_VARIANT \
                                --vjepa_img_size $VJEPA_IMG_SIZE \
                                --vjepa_masking_mode $VJEPA_MASKING_MODE \
                                --loss_mode "$LOSS_MODE" \
                                --dsearch_num_beams $NUM_BEAMS \
                                --dsearch_branch_k $BRANCH_K \
                                --dsearch_stride $STRIDE &
                        done
                        wait  # Wait for all GPUs on this node to complete this parameter combination
                    done
                    done
                    done
                
                elif [[ "$METHOD" == "svdd" ]]; then
                    # SVDD parameter combinations
                    for BRANCH_K in "${SVDD_BRANCH_K_LIST[@]}"; do
                    for BETA in "${SVDD_BETA_LIST[@]}"; do
                    for STRIDE in "${SVDD_STRIDE_LIST[@]}"; do
                        # Launch one worker per GPU on this node
                        for ((g=0; g<NUM_GPUS_PER_NODE; g++)); do
                            GLOBAL_GPU_IDX=$((NODE_ID * NUM_GPUS_PER_NODE + g))
                            echo "  -> Launching SVDD worker on Node $NODE_ID, Local GPU $g (Global GPU $GLOBAL_GPU_IDX), K=$BRANCH_K, beta=$BETA, stride=$STRIDE"
                            CUDA_VISIBLE_DEVICES=$g python generator_i2v_multinode.py \
                                --model_id "$MODEL_NAME" \
                                --output_folder "$RUN_OUTPUT_FOLDER" \
                                --batch_json "$BATCH_JSON" \
                                --base_dir "$BASEDIR" \
                                --num_gpus $TOTAL_GPUS \
                                --gpu_idx $GLOBAL_GPU_IDX \
                                --num_nodes $NUM_NODES \
                                --node_id $NODE_ID \
                                --gpus_per_node $NUM_GPUS_PER_NODE \
                                --sampling_method svdd \
                                --num_inference_steps $NUM_SAMPLING_STEPS \
                                --num_frames $NUM_FRAMES \
                                --height 480 \
                                --width 720 \
                                --cfg_scale $CFG_SCALE \
                                --vjepa_variant $VJEPA_VARIANT \
                                --vjepa_img_size $VJEPA_IMG_SIZE \
                                --vjepa_masking_mode $VJEPA_MASKING_MODE \
                                --loss_mode "$LOSS_MODE" \
                                --svdd_branch_k $BRANCH_K \
                                --svdd_beta $BETA \
                                --svdd_stride $STRIDE \
                                --config_version "v1" \
                                --seed 42 &
                        done
                        wait  # Wait for all GPUs on this node to complete this parameter combination
                    done
                    done
                    done
                fi
            done
        done
        done
    done
done
done

echo "Node ${NODE_ID} Search experiments completed! Results saved to: $OUTPUT_FOLDER"
