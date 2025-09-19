#!/bin/bash

# SLURM job array configuration for multi-node execution (SMC steering)
#SBATCH --job-name=cogvideox_smc
#SBATCH --array=0-0                    # nodes 0..2
#SBATCH --nodes=1                      # Each job uses 1 node
#SBATCH --qos=dev
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --gres=gpu:8                  # 8 GPUs per node
#SBATCH --cpus-per-task=48            # Adjust based on your cluster
#SBATCH --mem=512G                    # Adjust based on your cluster
#SBATCH --time=24:00:00               # Adjust based on expected runtime
#SBATCH --output=./jobs/smc_node_%A_%a.out
#SBATCH --error=./jobs/smc_%A_%a.err

source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
conda activate vg
nvidia-smi

# Multi-node configuration
NUM_NODES=1                           # Total number of nodes
NUM_GPUS_PER_NODE=8                   # GPUs per node
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
NODE_ID=${SLURM_ARRAY_TASK_ID}


echo "Starting node ${NODE_ID} of ${NUM_NODES} (GPUs per node: ${NUM_GPUS_PER_NODE}, Total GPUs: ${TOTAL_GPUS})"

# Sliding-window triplets (slice_window_size, vjepa_context_frames, slice_stride)
TRIPLETS=(
    "16 8 8"
)

# SMC parameters
SMC_NUM_PARTICLES_LIST=(2 12 16)
SMC_BETA_CONST=24.0
SMC_ESS_THRESHOLD=0.97
SMC_EARLY_FRAC=0.2
SMC_LATE_FRAC=0.8
SMC_STEP_STRIDE=3

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
    for triplet in "${TRIPLETS[@]}"; do
        read -r SLICE_WINDOW_SIZE CONTEXT_LENGTH STRIDE <<< "$triplet"
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

                # Loop over particle counts; launch one worker per GPU on this node
                for N_PARTICLES in "${SMC_NUM_PARTICLES_LIST[@]}"; do
                    for ((g=0; g<NUM_GPUS_PER_NODE; g++)); do
                        GLOBAL_GPU_IDX=$((NODE_ID * NUM_GPUS_PER_NODE + g))
                        echo "  -> Launching SMC worker on Node $NODE_ID, Local GPU $g (Global GPU $GLOBAL_GPU_IDX), N=$N_PARTICLES"
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
                            --sampling_method smc \
                            --num_inference_steps $NUM_SAMPLING_STEPS \
                            --num_frames $NUM_FRAMES \
                            --height 480 \
                            --width 720 \
                            --cfg_scale $CFG_SCALE \
                            --vjepa_variant $VJEPA_VARIANT \
                            --vjepa_img_size $VJEPA_IMG_SIZE \
                            --vjepa_masking_mode $VJEPA_MASKING_MODE \
                            --vjepa_context_frames $CONTEXT_LENGTH \
                            --slice_stride $STRIDE \
                            --slice_window_size $SLICE_WINDOW_SIZE \
                            --loss_mode "$LOSS_MODE" \
                            --smc_num_particles $N_PARTICLES \
                            --smc_beta_const $SMC_BETA_CONST \
                            --smc_ess_threshold $SMC_ESS_THRESHOLD \
                            --smc_early_frac $SMC_EARLY_FRAC \
                            --smc_late_frac $SMC_LATE_FRAC \
                            --smc_step_stride $SMC_STEP_STRIDE &
                    done
                    wait
                done
            done
            done
        done
    done
done
done

echo "Node ${NODE_ID} SMC experiments completed! Results saved to: $OUTPUT_FOLDER"
