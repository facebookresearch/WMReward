# Inference-time Physics Alignment of Video Generative Models with Latent World Models

## Installation

1. Clone the repo with submodules (vjepa2, MAGI-1)
```bash
git clone --recurse-submodules https://github.com/your-org/WMReward.git
cd WMReward
```

If you already cloned without `--recurse-submodules`, initialize submodules with:
```bash
git submodule update --init --recursive
git submodule sync --recursive
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
    --config_file ./MAGI-1/configs/inference_config.json \
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
