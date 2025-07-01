cd #!/bin/bash

# Script to evaluate PyTorch V-JEPA V2 on IntPhys dataset using corrected JEPA methodology
# This evaluates each context length separately and reports results like the original paper

set -e

# Configuration
INTPHYS_ROOT="/home/yjianhao/project/video_guidance"
SCRIPT_PATH="./reproduce_intphy_v2.py"
MODEL_PATH="/home/yjianhao/project/vjepa2/checkpoints/vitg-384.pt"
DEVICE="cuda"  # cuda, cpu
FRAME_STEP=2   # Original paper uses frame skip of 2 (matching original config)
FRAMES_PER_CLIP=16  # Standard for V-JEPA
IMG_SIZE=224   # Match original config (224, not 384!)
BATCH_SIZE=1

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    echo "Please make sure reproduce_intphy_v2.py is in the current directory"
    exit 1
fi

# Check if IntPhys data exists
if [ ! -d "$INTPHYS_ROOT" ]; then
    echo "Error: IntPhys data not found at $INTPHYS_ROOT"
    echo "Please update INTPHYS_ROOT in this script to point to your IntPhys data"
    exit 1
fi

# Check if model checkpoint exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at $MODEL_PATH"
    echo "Available checkpoints:"
    ls -la /home/yjianhao/project/vjepa2/checkpoints/
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

echo "================================================"
echo "PyTorch V-JEPA V2 IntPhys Evaluation (Corrected JEPA)"
echo "================================================"
echo "Data root: $INTPHYS_ROOT"
echo "Model: $MODEL_PATH"
echo "Frame skip: $FRAME_STEP"
echo "Frames per clip: $FRAMES_PER_CLIP"
echo "Image size: $IMG_SIZE"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
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
        --model_path "$MODEL_PATH" \
        --frame_step $FRAME_STEP \
        --frames_per_clip $FRAMES_PER_CLIP \
        --img_size $IMG_SIZE \
        --batch_size $BATCH_SIZE
    
    echo "Completed $parent_name/$dataset_name"
    echo ""
}

# Evaluate all dev datasets
echo "Starting evaluation of dev datasets..."
echo ""

run_evaluation "$INTPHYS_ROOT/dev/O1"
run_evaluation "$INTPHYS_ROOT/dev/O2" 
run_evaluation "$INTPHYS_ROOT/dev/O3"

echo "================================================"
echo "All PyTorch V-JEPA V2 evaluations completed!"
echo "================================================"
echo ""
echo "Results saved in timestamped directories:"
echo "- intphys_results_v2_pytorch_dev_O*_f${FRAMES_PER_CLIP}_s${FRAME_STEP}_YYYYMMDD_HHMMSS/"
echo ""
echo "Each directory contains:"
echo "- performance_pytorch_vjepa_v2.csv: Results in original paper format"
echo "- raw_losses_pytorch_vjepa_v2.pth: Raw loss data for further analysis"
echo "- detailed_metrics_pytorch_vjepa_v2.json: Detailed metrics breakdown"
echo ""
echo "CSV format matches original paper with columns:"
echo "- Block, Context length(s), Frame skip"
echo "- Relative Accuracy (avg), Relative Accuracy (max)"
echo "- Absolute Accuracy (max), Best Absolute Accuracy (max)"
echo "- AUPRC (avg), AUPRC (max), AUROC (avg), AUROC (max)" 
echo ""
echo "Note: Currently using simplified approach (single encoder) until"
echo "full JEPA checkpoint with target_encoder and predictor is available." 