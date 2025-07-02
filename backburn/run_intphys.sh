#!/bin/bash

# Script to evaluate V-JEPA on IntPhys dataset using the original paper's exact methodology
# This evaluates each context length separately and reports results like the original paper

set -e

# Configuration
INTPHYS_ROOT="/home/yjianhao/project/video_guidance"
SCRIPT_PATH="./reproduce_intphys.py"
DEVICE="auto"  # auto, cuda, cpu
FRAME_STEP=2   # Original paper uses frame skip of 2
FRAMES_PER_CLIP=16  # Standard for V-JEPA

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    echo "Please make sure reproduce_intphys.py is in the current directory"
    exit 1
fi

# Check if IntPhys data exists
if [ ! -d "$INTPHYS_ROOT" ]; then
    echo "Error: IntPhys data not found at $INTPHYS_ROOT"
    echo "Please update INTPHYS_ROOT in this script to point to your IntPhys data"
    exit 1
fi

echo "============================================"
echo "V-JEPA IntPhys Evaluation (Original Methodology)"
echo "============================================"
echo "Data root: $INTPHYS_ROOT"
echo "Frame skip: $FRAME_STEP"
echo "Frames per clip: $FRAMES_PER_CLIP"
echo "Device: $DEVICE"
echo ""

# Function to run evaluation on a dataset
run_evaluation() {
    local dataset_path=$1
    local dataset_name=$(basename "$dataset_path")
    local parent_name=$(basename "$(dirname "$dataset_path")")
    
    echo "Evaluating $parent_name/$dataset_name..."
    
    if [ ! -d "$dataset_path" ]; then
        echo "Warning: Dataset not found at $dataset_path, skipping..."
        return
    fi
    
    # Run evaluation
    python3 "$SCRIPT_PATH" \
        --data_path "$dataset_path" \
        --frame_step $FRAME_STEP \
        --frames_per_clip $FRAMES_PER_CLIP \
        --device "$DEVICE" \
        --batch_size 1
    
    echo "Completed $parent_name/$dataset_name"
    echo ""
}

# Evaluate all dev datasets
echo "Starting evaluation of dev datasets..."
echo ""

run_evaluation "$INTPHYS_ROOT/dev/O1"
run_evaluation "$INTPHYS_ROOT/dev/O2" 
run_evaluation "$INTPHYS_ROOT/dev/O3"

echo "============================================"
echo "All evaluations completed!"
echo "============================================"
echo ""
echo "Results saved in timestamped directories:"
echo "- intphys_results_dev_O*_f${FRAMES_PER_CLIP}_s${FRAME_STEP}_YYYYMMDD_HHMMSS/"
echo ""
echo "Each directory contains:"
echo "- performance.csv: Results in original paper format"
echo "- raw_losses.pth: Raw loss data for further analysis"
echo "- detailed_metrics.json: Detailed metrics breakdown"
echo ""
echo "CSV format matches original paper with columns:"
echo "- Block, Context length(s), Frame skip"
echo "- Relative Accuracy (avg), Relative Accuracy (max)"
echo "- Absolute Accuracy (max), Best Absolute Accuracy (max)"
echo "- AUPRC (max), AUROC (max)" 