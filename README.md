# Inference-time Physics Alignment of Video Generative Models with Latent World Models

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

For MAGI-1, follow instructions from [MAGI-1](https://github.com/SandAI-org/MAGI-1/tree/main) to install required dependency.

## Usage



### Compute VJEPA Surprise Reward
```bash
python compute_wmreward.py --video_path /path/to/video.mp4
```

Options:
- `--model`: Model variant (`vith`, `vitg`, `vitg384`, `vitgac`). Default: `vitg`
- `--window_size`: Sliding window size. Default: `16`
- `--context_frames`: Context frames per window. Default: `8`
- `--stride`: Sliding window stride. Default: `2`

### Quick Start (Single Prompt I2V)
```bash
python generate_cogvideox.py
```

Options:

**Input/Output:**
- `--prompt`: Text prompt describing the video. Default: example physics prompt
- `--init_image`: Path to initial image for I2V generation. Default: `./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg`

**WMReward Guidance:**
- `--guidance_step_pattern`: Step pattern for guidance (e.g., `0x3,1x47` = skip first 3 steps, apply for next 47). Default: `0x3,1x47`
- `--guidance_lr_pattern`: Learning rate pattern for guidance. Default: `0.003x50`
- `--guidance_frequency`: Frequency of guidance application. Default: `1`
- `--travel_time`: Travel time range. Default: `0,0`

**VJEPA Configuration:**
- `--vjepa_variant`: Model variant (`vit_large`, `vit_huge`, `vit_giant`, `vit_giant_384`). Default: `vit_giant`
- `--vjepa_img_size`: Input image size for VJEPA. Default: `256`
- `--vjepa_context_frames`: Number of context frames. Default: `8`
- `--slice_window_size`: Sliding window size. Default: `16`
- `--slice_stride`: Sliding window stride. Default: `8`
- `--vae_decode_scale`: VAE decode scale factor. Default: `0.8`

## Generate PhysicsIQ
Please follow the instructions from [PhysicsIQ](https://github.com/google-deepmind/physics-IQ-benchmark) to prepare the condition image and prompts. The prompt lists are provided in the `prompt` folder.


## Acknowledgements

Thanks to these great repositories: [MAGI-1](https://github.com/SandAI-org/MAGI-1/tree/main), [FrameGuidance](https://github.com/agwmon/frame-guidance) and many other inspiring works in the community.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
xxx
```
