#!/bin/bash
#
# Sora2 Physics-IQ T2V Generation
#
# This script generates videos for the Physics-IQ benchmark using Sora2 API.
# 198 prompts × 10 samples = 1,980 videos
#

SCRIPT_DIR="/home/reyhaneaskari/WMReward"
cd "$SCRIPT_DIR"

# =============================================================================
# Configuration
# =============================================================================
NUM_WORKERS=4                    # Parallel API workers
NUM_SAMPLES=10                   # Samples per prompt (for BoN)
SECONDS_DUR=8                    # Video duration (4, 8, or 12)
SIZE="1280x720"                  # Video size

BATCH_JSON="${SCRIPT_DIR}/prompts/physics_iq.json"
OUTPUT_FOLDER="${SCRIPT_DIR}/generated_videos/physics_iq/sora"
LOG_DIR="${SCRIPT_DIR}/logs"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "$LOG_DIR"
mkdir -p "${OUTPUT_FOLDER}"

TOTAL_ENTRIES=$(python3 -c "import json; print(len(json.load(open('${BATCH_JSON}'))))")
TOTAL_VIDEOS=$((TOTAL_ENTRIES * NUM_SAMPLES))

echo "============================================================"
echo "SORA2 PHYSICS-IQ T2V GENERATION"
echo "============================================================"
echo "Prompts:            ${TOTAL_ENTRIES}"
echo "Samples per prompt: ${NUM_SAMPLES}"
echo "Total videos:       ${TOTAL_VIDEOS}"
echo "Duration:           ${SECONDS_DUR}s"
echo "Size:               ${SIZE}"
echo "Workers:            ${NUM_WORKERS}"
echo "Output:             ${OUTPUT_FOLDER}"
echo "============================================================"
echo ""

# =============================================================================
# Launch parallel workers
# =============================================================================
echo "Launching ${NUM_WORKERS} workers..."
echo ""

PIDS=""
for w in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "[Worker ${w}] Starting..."
    
    nohup python3 -u generator_sora_physicsiq.py \
        --batch_json "${BATCH_JSON}" \
        --output_folder "${OUTPUT_FOLDER}" \
        --num_samples ${NUM_SAMPLES} \
        --seconds ${SECONDS_DUR} \
        --size "${SIZE}" \
        --num_workers ${NUM_WORKERS} \
        --worker_idx ${w} \
        >> "${LOG_DIR}/sora_worker_${w}.log" 2>&1 &
    
    PIDS="$PIDS $!"
    sleep 1
done

echo ""
echo "All workers started."
echo "PIDs:${PIDS}"
echo ""
echo "Logs: ${LOG_DIR}/sora_worker_*.log"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/sora_worker_0.log"
echo ""
echo "Waiting for all workers to complete..."
echo ""

# =============================================================================
# Wait for completion
# =============================================================================
for pid in $PIDS; do
    wait $pid || echo "Worker $pid exited with error"
done

# =============================================================================
# Summary
# =============================================================================
GENERATED=$(find "${OUTPUT_FOLDER}" -name "*.mp4" 2>/dev/null | wc -l)

echo ""
echo "============================================================"
echo "GENERATION COMPLETE"
echo "============================================================"
echo "Generated videos: ${GENERATED} / ${TOTAL_VIDEOS}"
echo "Output:           ${OUTPUT_FOLDER}"
echo "============================================================"
