#!/bin/bash
#SBATCH --job-name=physics_iq
#SBATCH --partition=h200
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=jobs/physics_iq_%j.out
#SBATCH --error=jobs/physics_iq_%j.err

#
# Run Physics-IQ evaluation on best-of-N=16 videos
#

# Activate conda env
source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
export PATH="/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin:$PATH"

cd /checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark

INPUT_FOLDER="/home/reyhaneaskari/WMReward/generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16_5s_best_exact"
OUTPUT_DIR="/home/reyhaneaskari/WMReward/results/physics_iq/n16_5s_best_exact"
DESCRIPTIONS_FILE="/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark/descriptions/descriptions.csv"

echo "============================================================"
echo "Physics-IQ Evaluation"
echo "============================================================"
echo "Input: $INPUT_FOLDER"
echo "Output: $OUTPUT_DIR"
echo ""

# Check input
NUM_VIDEOS=$(ls "$INPUT_FOLDER"/*.mp4 2>/dev/null | wc -l)
echo "Number of input videos: $NUM_VIDEOS"
echo ""

mkdir -p "$OUTPUT_DIR"

# Run Physics-IQ
python code/run_physics_iq.py \
    --input_folders "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_DIR" \
    --descriptions_file "$DESCRIPTIONS_FILE"

echo ""
echo "============================================================"
echo "Complete!"
echo "============================================================"
echo "Results: $OUTPUT_DIR"

