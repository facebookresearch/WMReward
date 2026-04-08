# Inference-time Physics Alignment of Video Generative Models with Latent World Models

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

2. Create conda environment (includes both WMReward and MAGI-1 dependencies)
```bash
conda env create -f environment.yml
conda activate wmreward
```

Alternatively, install manually:
```bash
conda create -n wmreward python=3.10
conda activate wmreward
pip install -r requirements.txt
pip install -r MAGI-1/requirements.txt
```

3. Download MAGI-1 model weights

Follow the instructions in the [MAGI-1 README](https://github.com/SandAI-org/MAGI-1#download) to download model weights. Place them so the directory structure looks like:
```
WMReward/
└── downloads/
    └── <MAGI-1 weight files>
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
