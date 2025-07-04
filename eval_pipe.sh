DIMENSIONS=('subject_consistency' 'temporal_flickering' 'aesthetic_quality' 'dynamic_degree' 'imaging_quality' 'motion_smoothness')

# Auto-discover models from generated videos directory
PROMPT='subject_consistency'
# Specify the list of experiments (model folders) to evaluate
EXPS=(
    # "guidance_f33_s50_w16c8_rho100.0_cfg5.0torch"
    # "guidance_f33_s50_w16c8_rho50.0_cfg5.0torch"
    # "guidance_f33_s50_w16c8_rho3.0_cfg5.0torch"
    # "guidance_f33_s50_w16c8_rho1.0_cfg5.0torch"
    # "vanilla_f33_s50_cfg5.0_0701_2103"
    # "vanilla_f33_s50_cfg5.0"
    # "guidance_f33_s50_w16c8_rho10.0_cfg5.0torch"
    "guidance_f33_s50_w16c8_rho1.0_cfg5.0nodyna"
    "guidance_f33_s50_w16c8_rho5.0_cfg5.0nodyna"
    "guidance_f33_s50_w16c8_rho10.0_cfg5.0nodyna"
    "guidance_f33_s50_w16c8_rho50.0_cfg5.0nodyna"
)
GENERATED_VIDEOS_DIR="../video_guidance/generated_videos/$PROMPT"
MODELS=("${EXPS[@]}")


if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models with video files found in $GENERATED_VIDEOS_DIR"
    echo "Please run generate.sh first!"
    exit 1
fi

echo "Discovered ${#MODELS[@]} models: ${MODELS[@]}"

source ~/miniconda3/etc/profile.d/conda.sh
cd ../VBench
conda activate vbench
echo "activated vbench environment"

# Get the number of available GPUs
NUM_GPUS=8
# Initialize a counter for GPU assignment
GPU_INDEX=0


for MODEL in "${MODELS[@]}"; do
    for DIMENSION in "${DIMENSIONS[@]}"; do
        echo "Evaluating dimension: $DIMENSION for model: $MODEL"
        VIDEO_PATH="../video_guidance/generated_videos/$PROMPT/$MODEL"
        OUT_PATH="../video_guidance/results/vbench/$PROMPT/$MODEL/$DIMENSION"
        mkdir -p "$OUT_PATH"
        
        # Assign the current task to a GPU
        CUDA_VISIBLE_DEVICES=$((GPU_INDEX % NUM_GPUS)) python3 ./evaluate.py \
            --dimension "$DIMENSION" \
            --videos_path "$VIDEO_PATH" \
            --output_path "$OUT_PATH" \
            --mode=custom_input &
        # Increment the GPU index
        GPU_INDEX=$((GPU_INDEX + 1))
    done
    wait
done
# Wait for all background processes to finish
wait

cd ../video_guidance

for MODEL in "${MODELS[@]}"
do
    echo "Generating test files for model: $MODEL"
    python3 ./scripts/make_videophy_testfile.py --model ${MODEL} --prompt ${PROMPT}
    mkdir -p /home/yjianhao/project/video_guidance/results/videophy/${MODEL}
done

cd ../videophy/VIDEOPHY2

# Activate the videophy environment
conda activate videophy
echo "Activated videophy environment"

# Initialize a counter for GPU assignment
GPU_INDEX=0

for MODEL in "${MODELS[@]}"; do
    echo "Running inference for model: $MODEL"

    # Assign the first task to a GPU
    CUDA_VISIBLE_DEVICES=$((GPU_INDEX % NUM_GPUS)) python inference.py \
        --input_csv /home/yjianhao/project/video_guidance/temp/$MODEL.csv \
        --checkpoint ./videophy_2_auto \
        --output_csv /home/yjianhao/project/video_guidance/results/videophy/${MODEL}/sa.csv \
        --task sa &

    # Increment the GPU index
    GPU_INDEX=$((GPU_INDEX + 1))

    # Assign the second task to a GPU
    CUDA_VISIBLE_DEVICES=$((GPU_INDEX % NUM_GPUS)) python inference.py \
        --input_csv /home/yjianhao/project/video_guidance/temp/$MODEL.csv \
        --checkpoint ./videophy_2_auto \
        --output_csv /home/yjianhao/project/video_guidance/results/videophy/${MODEL}/pc.csv \
        --task pc &

    # Increment the GPU index
    GPU_INDEX=$((GPU_INDEX + 1))
done

# Wait for all background processes to finish
wait