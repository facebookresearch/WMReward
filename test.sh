PROMPT="A ball falls from the table onto the floor"
CONFIG="./MAGI-1/example/24B/24B_base_config.json"
INIT="./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg"
OUT="./results/guidance_check"

mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=8 generate_magi1.py \
  --config_file "$CONFIG" \
  --prompt "$PROMPT" \
  --init_image "$INIT" \
  --output_path "$OUT/baseline.mp4" \
  --mode i2v \
  --guidance_scale 0 \
  --guidance_frequency 5

torchrun --standalone --nproc_per_node=8 generate_magi1.py \
  --config_file "$CONFIG" \
  --prompt "$PROMPT" \
  --init_image "$INIT" \
  --output_path "$OUT/guided.mp4" \
  --mode i2v \
  --guidance_scale 0.01 \
  --guidance_frequency 5

python3 tests/check_guidance_effect.py \
  --baseline "$OUT/baseline.mp4" \
  --guided "$OUT/guided.mp4"
