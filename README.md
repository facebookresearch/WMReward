# Inference-time Physics Alignment of Video Generative Models with Latent World Models

[![CVPR 2026 Highlight](https://img.shields.io/badge/CVPR%202026-Highlight-blue)](https://cvpr.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.10553-b31b1b.svg)](https://arxiv.org/abs/2601.10553)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://facebookresearch.github.io/WMReward/)
[![Video](https://img.shields.io/badge/Video-YouTube-red)](https://www.youtube.com/watch?v=NQzKR3xqU10)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](./LICENSE)

**[Paper](https://arxiv.org/pdf/2601.10553)** · **[arXiv](https://arxiv.org/abs/2601.10553)** · **[Project Page](https://facebookresearch.github.io/WMReward/)** · **[5-min Video](https://www.youtube.com/watch?v=NQzKR3xqU10)**

> **TL;DR.** A pretrained latent world model (VJEPA-2) is a strong physics-plausibility reward. We use it at inference time — no fine-tuning of the generator — to steer video diffusion toward physically plausible samples, achieving state-of-the-art on the PhysicsIQ benchmark across MAGI-1, Sora 2, and a video latent diffusion model.

**Headline results on the PhysicsIQ benchmark:** +5.68 on vLDM · +4.13 on Sora 2 · +6.78 on MAGI-1 · **62.64%** final score — new state of the art, +7.42 over Sora 2. First place in the ICCV 2025 Perception Test PhysicsIQ Challenge.

---

## Installation

1. Clone the repo with submodules (vjepa2, MAGI-1)
```bash
git clone --recurse-submodules https://github.com/facebookresearch/WMReward.git
cd WMReward
```

If you already cloned without `--recurse-submodules`, initialize submodules with:
```bash
git submodule update --init --recursive
git submodule sync --recursive
```

2. Create conda environment and install dependencies (Python 3.10 + PyTorch 2.4 with CUDA 12.4)
```bash
conda env create -f environment.yml
conda activate wmreward
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.4.2 --no-build-isolation
pip install flashinfer-python==0.2.0.post2 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/
```

3. Download MAGI-1 model weights (only needed for video generation, not for `compute_wmreward.py`)

Download from the [MAGI-1 Hugging Face repo](https://huggingface.co/sand-ai/MAGI-1):
```bash
pip install "huggingface_hub[cli]"

# Download the 24B base model, VAE, and T5 text encoder
huggingface-cli download sand-ai/MAGI-1 --include "ckpt/magi/24B_base/*" --local-dir downloads
huggingface-cli download sand-ai/MAGI-1 --include "ckpt/vae/*" --local-dir downloads
huggingface-cli download sand-ai/MAGI-1 --include "ckpt/t5/*" --local-dir downloads

# Move into the expected layout
mv downloads/ckpt/magi/24B_base downloads/24B_base
mv downloads/ckpt/vae downloads/vae
mv downloads/ckpt/t5 downloads/t5_pretrained
rm -rf downloads/ckpt
```

The expected directory structure:
```
WMReward/
└── downloads/
    ├── 24B_base/       # MAGI-1 DiT model weights
    ├── vae/            # MAGI-1 VAE encoder/decoder
    └── t5_pretrained/  # T5-XXL text encoder
```

> **Note:** VJEPA checkpoints are **optional** for computing WMReward. The `compute_wmreward.py` script automatically downloads them via `torch.hub`. If you want to use local checkpoints (via `load_vjepa_model_source`), place them in `./checkpoints/` or set `VJEPA_CHECKPOINT_DIR` to your checkpoint directory.

## Usage

### Compute VJEPA Surprise Reward
Our WMReward is computed with the central function compute_vjepa_surprise() currently implemented for VJEPA models.
```bash
python compute_wmreward.py --video_path /path/to/video.mp4
```

Options:
- `--model`: Model variant (`vith`, `vitg`, `vitg384`, `vitgac`). Default: `vitg`
- `--window_size`: Sliding window size. Default: `16`
- `--context_frames`: Context frames per window. Default: `8`
- `--stride`: Sliding window stride. Default: `2`

Other models can be pretty easily integrated. Just compute a reward score with them, e.g. a yes/no log likelihood with a VLM. For WMReward Guidance on your own model, you can also use this function. We implemented the guidance too for MAGI-1 in `generator_i2v_multinode.py`.

### Quick Start (Single Prompt I2V)
```bash
python generate_magi1.py \
    --config_file ./MAGI-1/example/24B/24B_base_config.json \
    --prompt "A ball falls from the table onto the floor" \
    --init_image ./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg \
    --output_path ./results/output.mp4 \
    --mode i2v
```

Options:

**Input/Output:**
- `--prompt`: Text prompt describing the video (required)
- `--config_file`: Path to MAGI-1 configuration JSON file (required)
- `--output_path`: Path to save the output video (required)
- `--mode`: Generation mode: `t2v` (text-to-video), `i2v` (image-to-video), `v2v` (video-to-video). Default: `i2v`
- `--init_image`: Path to initial image for I2V mode
- `--init_video`: Path to prefix video for V2V mode

## Generate PhysicsIQ
Please follow the instructions from [PhysicsIQ](https://github.com/google-deepmind/physics-IQ-benchmark) to prepare the condition image and prompts. The prompt lists are provided in the `prompt` folder. Then run
```bash
bash generation/generate_i2v_magi1_multinode.sh
```


## Acknowledgements

Thanks to these great repositories: [MAGI-1](https://github.com/SandAI-org/MAGI-1/tree/main), [FrameGuidance](https://github.com/agwmon/frame-guidance) and many other inspiring works in the community.

## License

This project is licensed under the CC BY-NC 4.0 License - see the [LICENSE](LICENSE) file for details. Whenever we make use of other repos (MAGI-1 and VJEPA) those fall under their own copyright and license. Please make sure you adhere to them too.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{yuan2026inferencetimephysicsalignmentvideo,
      title={Inference-time Physics Alignment of Video Generative Models with Latent World Models},
      author={Jianhao Yuan and Xiaofeng Zhang and Felix Friedrich and Nicolas Beltran-Velez and Melissa Hall and Reyhane Askari-Hemmat and Xiaochuang Han and Nicolas Ballas and Michal Drozdzal and Adriana Romero-Soriano},
      year={2026},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
