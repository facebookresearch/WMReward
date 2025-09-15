
### Align Video Diffusion Model to Latent World Model

Minimal tooling to run image-to-video (I2V) generation with three modes:
- vanilla: baseline generation
- guidance: V-JEPA surprise guidance during sampling
- rejection: generate k candidates, score with V-JEPA, keep best

Supports both CogVideoX and NVIDIA Cosmos I2V checkpoints via customized pipelines in `pipelines/`.

### Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
- Install Python deps:
```bash
pip install -r requirements.txt
```

### Quick start (single prompt)
Use these script entrypoints for quick local runs.

- CogVideoX:
```bash
python ./run_i2v_cogvideox.py \
  --init_image ./path/to/image.png \
  --prompt "A robot hand picks up a cup." \
  --model_id THUDM/CogVideoX-5b-I2V \
  --num_frames 49 --steps 50 --guidance_scale 6.0 \
  --guidance_step_pattern "0x5,1x45" \
  --guidance_lr_pattern "0.005x50" \
  --out_dir ./results/i2v_vjepa_slicepred --run_name demo
```

- Cosmos:
Edit `run_i2v_cosmos.py` to set your `image = load_image("./path/to/image.png")` and `prompt = "..."` lines, then run:
```bash
python ./run_i2v_cosmos.py \
  --mode guidance \
  --cfg 7.0 \
  --guidance_step_pattern "0x3,3x12,2x12,1x23" \
  --guidance_lr_pattern "3.0x15,2.0x15,1.0x20" \
  --guidance_freq 1 \
  --seed 42
```


### Batch mode (JSON) — use bash scripts
Provide a JSON array with entries. Each item must have either `input_image` or `input_video`, plus `prompt` and `output_video`.
```json
[
  {
    "input_image": "relative/or/absolute/path.png",
    "prompt": "Use the left hand to pick up the cucumber.",
    "output_video": "object/left_hand_pickup.mp4"
  }
]
```
Run batch generation by editing the bash scripts and executing them:
- Single node, all GPUs:
```bash
# Edit variables inside the script (MODEL_NAMES, BATCH_JSON_LIST, OUTPUT_FOLDER, SAMPLE_METHODS, etc.)
bash ./generate_i2v_cosmos.sh
```
- Multi-node SLURM (guidance):
```bash
# Edit NUM_NODES, MODEL_NAMES, BATCH_JSON_LIST, OUTPUT_FOLDER, etc.
sbatch ./generate_i2v_cosmos_multinode.sh
```
- Multi-node SLURM (rejection with buffer reuse):
```bash
# Edit REJECTION_SAMPLES and other variables as needed
sbatch ./generate_i2v_cosmos_rejection_multinode.sh
```

Notes:
- JSONs under Physics-IQ generally use absolute input paths; DreamBench JSONs are relative to `BASEDIR` set in the scripts.
- `prompts/physics_iq.json` is an example JSON you can add to a script’s `BATCH_JSON_LIST`.

### Multi-GPU / multi-node helpers
- Single node, all GPUs: edit and run
```bash
bash /home/yjianhao/project/video_guidance/generate_i2v_cosmos.sh
```
- SLURM multi-node (guidance):
```bash
sbatch /home/yjianhao/project/video_guidance/generate_i2v_cosmos_multinode.sh
```
- SLURM multi-node (rejection, with buffer reuse):
```bash
sbatch /home/yjianhao/project/video_guidance/generate_i2v_cosmos_rejection_multinode.sh
```
Before submitting, adjust inside the scripts: `MODEL_NAMES`, `BATCH_JSON_LIST`, `OUTPUT_FOLDER`, and node/GPU counts.

### Outputs
By default, videos are saved under:
- Single prompt: the provided `--output_path`
- Batch: `--output_folder/<experiment_name>/<prompt>.mp4`
Experiment folders include a simple CSV log and `experiment_config.json` for reproducibility.

### Useful files
- `generator_i2v.py`: single-node batch/single prompt runner
- `generator_i2v_multinode.py`: multi-node aware runner
- `generate_i2v_rejection.py`: rejection and rej_guide modes
- `pipelines/`: customized CogVideoX and Cosmos I2V pipelines
- `prompts/physics_iq.json`: example batch

