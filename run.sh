#!/bin/bash
#
# Physics-IQ Video Generation with Sora2
#
# Usage:
#   ./run.sh                    # Default: I2V mode, 4 workers
#   ./run.sh --mode t2v         # Text-to-Video mode
#   ./run.sh --workers 8        # Use 8 parallel workers
#

set -e
cd "$(dirname "$0")"

# ============================================================================
# CREDENTIALS - Set these before running
# ============================================================================
export SORA_HOST="azure-services-fair-openai2-eastus2n6.azure-api.net"
export SORA_API_KEY="a60a6c3d975747ef9866d1827c266976"

# ============================================================================
# Configuration
# ============================================================================
MODE="i2v"          # i2v (image-to-video) or t2v (text-to-video)
WORKERS=4           # Parallel API workers
SAMPLES=10          # Samples per prompt for Best-of-N
SECONDS=8           # Duration: 4, 8, or 12
SIZE="1280x720"     # Video resolution

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)     MODE="$2"; shift 2;;
        --workers)  WORKERS="$2"; shift 2;;
        --samples)  SAMPLES="$2"; shift 2;;
        --seconds)  SECONDS="$2"; shift 2;;
        --size)     SIZE="$2"; shift 2;;
        *)          echo "Unknown: $1"; exit 1;;
    esac
done

# ============================================================================
# Setup
# ============================================================================
mkdir -p logs

TOTAL=$(python3 -c "import json; print(len(json.load(open('prompts/physics_iq.json'))))")
echo ""
echo "=========================================="
echo "SORA2 PHYSICS-IQ ${MODE^^}"
echo "=========================================="
echo "Entries:  $TOTAL"
echo "Samples:  $SAMPLES per entry"
echo "Workers:  $WORKERS"
echo "Duration: ${SECONDS}s @ $SIZE"
echo "=========================================="
echo ""

# ============================================================================
# Launch workers
# ============================================================================
PIDS=""
for w in $(seq 0 $((WORKERS - 1))); do
    echo "Starting worker $w..."
    
    nohup python3 -u generator_sora_physicsiq.py \
        --mode "$MODE" \
        --num_samples "$SAMPLES" \
        --seconds "$SECONDS" \
        --size "$SIZE" \
        --num_workers "$WORKERS" \
        --worker_idx "$w" \
        >> "logs/sora_${MODE}_w${w}.log" 2>&1 &
    
    PIDS="$PIDS $!"
    sleep 0.5
done

echo ""
echo "Workers started: $PIDS"
echo "Logs: logs/sora_${MODE}_w*.log"
echo ""
echo "Monitor: tail -f logs/sora_${MODE}_w0.log"
echo ""

# Wait for all
for pid in $PIDS; do
    wait $pid || echo "Worker $pid failed"
done

echo ""
echo "=========================================="
echo "DONE"
echo "=========================================="
