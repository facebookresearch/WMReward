#!/bin/bash
#SBATCH --job-name=trim_exact
#SBATCH --partition=h200
#SBATCH --qos=h200_dream_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=jobs/trim_exact_%j.out
#SBATCH --error=jobs/trim_exact_%j.err

#
# Trim best-of-N videos to EXACTLY 5 seconds (re-encode for precision)
#

# Activate conda env with ffmpeg
source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
export PATH="/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin:$PATH"

cd /home/reyhaneaskari/WMReward

INPUT_DIR="./generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16_5s_best"
OUTPUT_DIR="./generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16_5s_best_exact"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Trimming videos to EXACTLY 5 seconds (with re-encoding)"
echo "============================================================"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check ffmpeg
echo "Checking ffmpeg..."
which ffmpeg
ffmpeg -version | head -1
echo ""

# Count input videos
TOTAL=$(ls "$INPUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Total videos to trim: $TOTAL"
echo ""

# Trim each video with re-encoding for exact duration
COUNT=0
SKIPPED=0
ERRORS=0

for VIDEO in "$INPUT_DIR"/*.mp4; do
    BASENAME=$(basename "$VIDEO")
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME"
    
    # Skip if already trimmed
    if [ -f "$OUTPUT_FILE" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Trim to exactly 5 seconds with re-encoding
    # Using -vframes 150 (30 FPS * 5s = 150 frames) ensures EXACT 5.000s duration
    # -r 30 ensures 30 FPS output, -an removes audio
    if ffmpeg -y -i "$VIDEO" -r 30 -vframes 150 -c:v libx264 -preset fast -crf 18 -an "$OUTPUT_FILE" 2>/dev/null; then
        COUNT=$((COUNT + 1))
    else
        ERRORS=$((ERRORS + 1))
        echo "ERROR: Failed to trim $BASENAME"
    fi
    
    # Progress every 20 videos
    if [ $(((COUNT + ERRORS) % 20)) -eq 0 ]; then
        echo "Progress: $COUNT trimmed, $ERRORS errors, $SKIPPED skipped / $TOTAL total"
    fi
done

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Trimmed: $COUNT"
echo "Skipped (already exist): $SKIPPED"
echo "Errors: $ERRORS"
echo "Output: $OUTPUT_DIR"

# Verify count
OUTPUT_COUNT=$(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Total output videos: $OUTPUT_COUNT / $TOTAL expected"

# Verify duration of first video
echo ""
echo "Verifying first video duration..."
FIRST_VIDEO=$(ls "$OUTPUT_DIR"/*.mp4 | head -1)
ffprobe -v error -show_entries format=duration -of csv=p=0 "$FIRST_VIDEO"

