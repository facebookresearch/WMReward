### WMReward: Align Video Diffusion Model to Latent World Model

Minimal tooling to run image/video-to-video (I2V/V2V) generation with CogVideoX and MAGI-1.

## Installation

1. Clone the repo with submodules (vjepa2, MAGI-1)
```bash
git clone --recurse-submodules https://github.com/fairinternal/video_guidance.git
cd WMReward
```

If you already cloned without `--recurse-submodules`, initialize submodules with:
```bash
git submodule update --init --recursive
```

2. Create conda environment
```bash
conda create -n vg python=3.12
conda activate vg
pip install torch torchvision
pip install diffusers transformers
pip install -r requirements.txt
```

For MAGI-1, follow instructions from [MAGI-1](https://github.com/SandAI-org/MAGI-1/tree/main)

## Usage

### Quick Start (Single Prompt)
```bash
python run_i2v_cogvideox.py
```

### Compute VJEPA World Model Reward
```bash
python compute_wmreward.py --video_path /path/to/video.mp4
```

Example with sample video:
```bash
python compute_wmreward.py --video_path /storage/home/yjianhao/project/MAGI-1/example/assets/output_i2v.mp4
```

Options:
- `--model`: Model variant (`vith`, `vitg`, `vitg384`, `vitgac`). Default: `vitg`
- `--window_size`: Sliding window size. Default: `16`
- `--context_frames`: Context frames per window. Default: `8`
- `--stride`: Sliding window stride. Default: `2`
- `--seed`: Random seed. Default: `42`

### Batch Generation (SLURM)
```bash
sbatch generation/generate_batch_1chunk_abl.sh
```

## Project Structure
```
.
├── evaluation/          # Evaluation scripts
├── example/             # Config files for generation
├── generated_videos/    # Output directory for generated videos
├── generation/          # Generation scripts (SLURM entry points)
├── inference/           # Inference pipelines
├── prompts/             # Generation prompts
├── scripts/             # Helper scripts
├── utils.py             # Utils for VJEPA score and video processing
├── compute_wmreward.py  # Compute VJEPA surprise score
├── vjepa2/              # V-JEPA2 submodule
└── MAGI-1/              # MAGI-1 submodule
```

## Pipeline Overview

**Generation**: `generation/` → `inference/generator.py` → `inference/pipeline_w_guidance.py` → `inference/video_generate_longcontext_skip_guidance.py`

**Evaluation**: `evaluation/eval_physics_iq_parallel.sh` → `scripts/physicsiq_res2tab.py`
