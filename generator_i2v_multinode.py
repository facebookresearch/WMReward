# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s

"""
MAGI-1 multi-node I2V generation with VJEPA guidance support.
This script handles batch video generation using MAGI-1's guidance pipeline,
with support for multi-GPU sharding and VJEPA-based rejection sampling.
"""

import argparse
import json
from diffusers.utils import export_to_video
try:
    from diffusers.utils import load_video
except ImportError:
    def load_video(path):
        from decord import VideoReader
        from PIL import Image as _Image
        vr = VideoReader(str(path))
        return [_Image.fromarray(frame.asnumpy()) for frame in vr]
from datetime import datetime
import torch
import cv2
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import math
import numpy as np
from PIL import Image
from utils import compute_vjepa_loss_sliding_window, load_vjepa_models_torchhub

# Add MAGI-1 submodule to path
MAGI1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MAGI-1")
if MAGI1_PATH not in sys.path:
    sys.path.insert(0, MAGI1_PATH)

# Set SPECIAL_TOKEN_PATH for MAGI-1 if not already set
os.environ.setdefault("SPECIAL_TOKEN_PATH", os.path.join(MAGI1_PATH, "example/assets/special_tokens.npz"))

def set_deterministic(seed=42):
    """Set deterministic behavior for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def parse_range_pair(text: str):
    a, b = (text.split(",", 1) if "," in text else text.split("-", 1))
    return int(a.strip()), int(b.strip())

def parse_float_range_pair(text: str):
    a, b = (text.split(",", 1) if "," in text else text.split("-", 1))
    return float(a.strip()), float(b.strip())


def normalize_vjepa_variant(model_name: str) -> str:
    alias_map = {
        "vith": "vith",
        "vit_huge": "vith",
        "vitg": "vitg",
        "vit_giant": "vitg",
        "vitg384": "vitg384",
        "vit_giant_384": "vitg384",
        "vitgac": "vitgac",
        "vit_giant_ac": "vitgac",
    }
    if model_name not in alias_map:
        raise ValueError(
            f"Unsupported V-JEPA variant '{model_name}'. "
            "Use one of: vith, vit_huge, vitg, vit_giant, vitg384, vit_giant_384, vitgac, vit_giant_ac."
        )
    return alias_map[model_name]


def guidance_metadata_only_args(args) -> dict:
    context_frames, stride, window_size = _resolve_sliding_window_params(args)
    return {
        "guidance_start": getattr(args, "guidance_start", None),
        "guidance_end": getattr(args, "guidance_end", None),
        "guidance_rho_scale": getattr(args, "guidance_rho_scale", None),
        "travel_time": getattr(args, "travel_time", None),
        "guidance_step_pattern": getattr(args, "guidance_step_pattern", None),
        "guidance_lr_pattern": getattr(args, "guidance_lr_pattern", None),
        "vjepa_context_frames": context_frames,
        "slice_stride": stride,
        "slice_window_size": window_size,
        "loss_mode": getattr(args, "loss_mode", None),
    }

def save_experiment_metadata(args, experiment_name, experiment_folder):
    """Save experiment metadata as JSON file in the experiment folder."""

    # Create metadata with all relevant parameters
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config_version": getattr(args, 'config_version', 'v2'),
        "parameters": {
            "sampling_method": args.sampling_method,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "config_file": args.config_file,
            "prompt_file": getattr(args, 'prompt_file', None),
            "batch_json": getattr(args, 'batch_json', None),
            "height": getattr(args, 'height', 480),
            "width": getattr(args, 'width', 720),
        }
    }

    # Add guidance-specific parameters
    if args.sampling_method == 'guidance':
        metadata["parameters"].update({
            "guidance_scale": getattr(args, 'guidance_scale', None),
            "guidance_frequency": getattr(args, 'guidance_frequency', None),
            "vjepa_type": getattr(args, 'vjepa_type', None),
            "vjepa_variant": getattr(args, 'vjepa_variant', None),
            "legacy_metadata_only_args": guidance_metadata_only_args(args),
        })

    # Add rejection sampling parameters
    if args.sampling_method == 'rejection':
        metadata["parameters"].update({
            "rejection_samples": getattr(args, 'rejection_samples', 3),
            "loss_mode": getattr(args, 'loss_mode', 'mean'),
        })

    # Save metadata file
    metadata_path = os.path.join(experiment_folder, "experiment_config.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved experiment metadata to: {metadata_path}")

def get_simple_experiment_name(args):
    """Generate a clean, short experiment folder name (with optional version tag)."""

    version = getattr(args, 'config_version', 'v2')

    # Base name with method and key params
    if args.sampling_method == 'vanilla':
        name = f"vanilla_{version}_f{args.num_frames}_s{args.num_inference_steps}_cfg{args.cfg_scale}_seed{args.seed}"
    elif args.sampling_method == 'guidance':
        vjepa_variant = getattr(args, 'vjepa_variant', 'vit_giant')
        # Simplify vjepa variant name for brevity
        vjepa_short = vjepa_variant.replace('vit_', '') if vjepa_variant != 'vit_giant' else ''

        name = (
            f"guidance_{version}_f{args.num_frames}_s{args.num_inference_steps}"
            f"_gs{getattr(args, 'guidance_scale', '')}_gf{getattr(args, 'guidance_frequency', '')}"
            f"_cfg{args.cfg_scale}_seed{args.seed}"
        )

        # Add vjepa variant if not default
        if vjepa_short:
            name += f"_{vjepa_short}"
    elif args.sampling_method == 'rejection':
        name = f"{args.sampling_method}_{version}_f{args.num_frames}_s{args.num_inference_steps}_cfg{args.cfg_scale}"
    # Add rejection samples suffix if using rejection sampling
    if args.sampling_method == 'rejection':
        name += f"_reject{getattr(args, 'rejection_samples', 3)}_{getattr(args, 'loss_mode', 'mean')}"

    if "5frame" in args.batch_json:
        name += "_5frame"

    # name+=f"_{args.seed}"

    return name

def _resolve_sliding_window_params(args):
    """Return unified (context_frames, stride, window_size) using single source of truth.
    Prefers slice-pred args; falls back to legacy names for compatibility.
    """
    context_frames = int(getattr(args, 'vjepa_context_frames', getattr(args, 'context_length', 8)))
    stride = int(getattr(args, 'slice_stride', getattr(args, 'stride', 4)))
    # For torch V-JEPA, window size is typically 16; prefer explicit slice_window_size, else kernel_size, else 16
    window_size = int(getattr(args, 'slice_window_size', getattr(args, 'kernel_size', 16)))
    return context_frames, stride, window_size

def log_experiment_simple(args, experiment_name, status='started'):
    """Simple logging to CSV."""
    log_file = os.path.join(args.output_folder, 'experiments.csv')

    # Create header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('name,method,frames,steps,context_frames,slice_stride,g_step_pattern,g_lr_pattern,g_frequency,travel_time,cfg_scale,timestamp,status\n')

    # Add entry
    with open(log_file, 'a') as f:
        context_frames, stride, _ = _resolve_sliding_window_params(args)
        f.write(f"{experiment_name},{args.sampling_method},{args.num_frames},{args.num_inference_steps},")
        f.write(f"{context_frames},{stride},")
        f.write(f"{getattr(args, 'guidance_step_pattern', '')},{getattr(args, 'guidance_lr_pattern', '')},")
        f.write(f"{getattr(args, 'guidance_frequency', '')},{getattr(args, 'travel_time', '')},")
        f.write(f"{args.cfg_scale},")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{status}\n")

def get_simple_output_folder(args, experiment_name):
    """Get simple output folder structure."""
    folder = os.path.join(args.output_folder, experiment_name)
    os.makedirs(folder, exist_ok=True)
    return folder

def find_existing_video_for_prompt(experiment_folder: str, prompt: str) -> str | None:
    """Return a path to an existing video in folder that matches *_<prompt>.mp4, else None."""
    if not os.path.isdir(experiment_folder):
        return None
    suffix = f"_{prompt}.mp4"
    try:
        for name in os.listdir(experiment_folder):
            if name.endswith(suffix):
                return os.path.join(experiment_folder, name)
    except FileNotFoundError:
        return None
    return None

def get_prompts(prompt_file, args):
    """Read prompts and negative prompts from a text file."""
    with open(f"./prompts/{prompt_file}.txt", 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]

    # Define a negative prompt for video generation
    negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"

    return prompts, negative_prompt

def load_first_frame(image_path: str | None, video_path: str | None) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    if not video_path:
        raise ValueError("Provide either --init_image or --init_video")

    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read from video: {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def init_pipeline(args):
    """Initialize the MAGI-1 pipeline with VJEPA guidance support."""
    from inference.pipeline.pipeline_w_guidance import MagiPipeline

    pipeline = MagiPipeline(args.config_file)
    pipeline.guidance_scale = getattr(args, "guidance_scale", pipeline.guidance_scale)
    pipeline.guidance_frequency = getattr(args, "guidance_frequency", pipeline.guidance_frequency)
    pipeline.vjepa_type = normalize_vjepa_variant(args.vjepa_type or args.vjepa_variant)
    return pipeline

def init_vjepa_models(args):
    """Initialize V-JEPA models for rejection sampling evaluation."""
    if args.sampling_method != 'rejection':
        return None, None, None

    normalized_variant = normalize_vjepa_variant(args.vjepa_variant)
    print(f"Loading V-JEPA models for rejection sampling ({normalized_variant})...")
    encoder, target_encoder, predictor, img_size = load_vjepa_models_torchhub(normalized_variant)
    encoder.eval().cuda()
    target_encoder.eval().cuda()
    predictor.eval().cuda()
    print(f"V-JEPA models loaded successfully (img_size: {img_size})")
    return encoder, target_encoder, predictor


def _build_seq(pattern: str, steps: int, is_float: bool):
    tokens = [p.strip() for p in pattern.split(",") if p.strip()]
    seq = []
    for tok in tokens:
        if "x" in tok:
            v, c = tok.split("x")
            seq.extend(([float(v) if is_float else int(v)]) * int(c))
        else:
            v = float(tok) if is_float else int(tok)
            seq = [v] * steps
            break
    if len(seq) != steps:
        raise ValueError(f"bad pattern len {len(seq)} vs {steps}")
    return seq

def guidance_sample(pipe, args, init_frame, prompt, negative_prompt, output_path, generator=None):
    """Guidance sampling using MAGI-1's built-in VJEPA guidance pipeline."""

    # Save init_frame temporarily for MAGI-1
    temp_image_path = "/tmp/magi1_init_frame.png"
    if isinstance(init_frame, Image.Image):
        init_frame.save(temp_image_path)
    else:
        Image.fromarray(init_frame).save(temp_image_path)

    # Use MAGI-1's I2V with built-in guidance
    pipe.run_image_to_video(
        prompt=prompt,
        image_path=temp_image_path,
        output_path=output_path
    )

    # Read the generated video frames back
    frames = load_video(output_path)
    return frames

def _ensure_btchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise RuntimeError(f"expected 5D video, got {x.shape}")
    if x.shape[1] == 3:
        return x
    if x.shape[2] == 3:
        return x.permute(0, 2, 1, 3, 4).contiguous()
    if x.shape[-1] == 3:
        return x.permute(0, 4, 1, 2, 3).contiguous()
    raise RuntimeError(f"Cannot infer channel dim in {x.shape}; expected channel==3 at dim 1/2/-1.")

def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    xmin = float(x.min())
    xmax = float(x.max())
    if -0.05 <= xmin and xmax <= 1.05:
        return x * 2.0 - 1.0
    if 0.0 <= xmin and xmax <= 255.0:
        return (x / 127.5) - 1.0
    return x

@torch.inference_mode()
def _vjepa_surprise_batch(vids_btchw: torch.Tensor, encoder, target_encoder, predictor, args) -> torch.Tensor:
    vids_btchw = _ensure_btchw(vids_btchw).to(dtype=torch.float32)
    B = vids_btchw.shape[0]
    out = torch.empty(B, device=vids_btchw.device, dtype=torch.float32)
    for i in range(0, B):
        loss = compute_vjepa_loss_sliding_window(
            video_tensor=vids_btchw[i:i+1],
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            img_size=args.vjepa_img_size,
            window_size=int(args.slice_window_size),
            loss_exp=2,
            masking_mode=str(args.vjepa_masking_mode),
            context_frames=int(args.vjepa_context_frames),
            spatial_pred_mask_scale=None,
            temporal_pred_mask_scale=None,
            aspect_ratio=None,
            npred=None,
            max_context_frames_ratio=None,
            is_vae_output=True,
            seed=int(args.seed),
            stride=int(args.slice_stride),
            mode=str(args.loss_mode),
        )
        out[i] = float(loss)
    return out


def generate_videos(pipe, args, init_frame, prompts, negative_prompt, experiment_name, fps=8, vjepa_models=None):
    """Generate videos for each prompt and save them to the output folder."""

    # Always ensure experiment folder exists; outputs default here unless output_path is explicitly used
    experiment_folder = get_simple_output_folder(args, experiment_name)

    # If explicit output path is provided for single-prompt mode, ensure its directory exists
    if getattr(args, 'output_path', None):
        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Log experiment start
    log_experiment_simple(args, experiment_name, 'started')

    # Save metadata in experiment folder
    save_experiment_metadata(args, experiment_name, experiment_folder)


    # Unpack V-JEPA models for rejection sampling
    encoder, target_encoder, predictor = vjepa_models if vjepa_models else (None, None, None)

    # Generate videos for each prompt
    for i, prompt in enumerate(prompts):
        safe_prompt = prompt
        # Build output path priority:
        # 1) --output_path for single prompt
        # 2) --output_filename for single prompt under <output_folder>/<experiment>
        # 3) Default: <output_folder>/<experiment>/<prompt>.mp4
        if getattr(args, 'output_path', None) and len(prompts) == 1:
            video_path = args.output_path
        elif getattr(args, 'output_filename', None) and len(prompts) == 1:
            video_path = os.path.join(experiment_folder, args.output_filename)
        else:
            # Match t2v naming: "<prompt>.mp4"
            video_path = os.path.join(experiment_folder, f"{safe_prompt}.mp4")
        print(video_path)
        print(os.path.exists(video_path))
        if os.path.exists(video_path):
            print(f"Video already exists, skipping: {video_path}")
            continue
        # Prompt-based existence check (ignore index prefix)
        existing_by_prompt = find_existing_video_for_prompt(experiment_folder, safe_prompt)
        if existing_by_prompt:
            print(f"Video for prompt already exists, skipping: {existing_by_prompt}")
            continue

        print(f"[{experiment_name}] Generating video {i+1}/{len(prompts)} ({args.sampling_method}): {prompt}")

    # Generate frames
        if args.sampling_method == 'vanilla':
            # Save init_frame temporarily for MAGI-1
            temp_image_path = "/tmp/magi1_init_frame.png"
            if isinstance(init_frame, Image.Image):
                init_frame.save(temp_image_path)
            else:
                Image.fromarray(init_frame).save(temp_image_path)

            # Use MAGI-1's I2V
            pipe.run_image_to_video(
                prompt=prompt,
                image_path=temp_image_path,
                output_path=video_path
            )

            # Read the generated video frames back for export
            frames = load_video(video_path)
        elif args.sampling_method == 'guidance':
            # Save init_frame temporarily for MAGI-1
            temp_image_path = "/tmp/magi1_init_frame.png"
            if isinstance(init_frame, Image.Image):
                init_frame.save(temp_image_path)
            else:
                Image.fromarray(init_frame).save(temp_image_path)

            # Use MAGI-1's I2V with built-in guidance
            pipe.run_image_to_video(
                prompt=prompt,
                image_path=temp_image_path,
                output_path=video_path
            )

            # Read the generated video frames back
            frames = load_video(video_path)


        elif args.sampling_method == 'rejection':
            print(f"  Generating with rejection sampling ({args.rejection_samples} candidates)...")

            # Generate multiple samples and select best based on V-JEPA loss
            candidate_frames = []
            candidate_losses = []

            for sample_idx in range(args.rejection_samples):
                print(f"    Generating candidate {sample_idx + 1}/{args.rejection_samples}...")

                # Save init_frame temporarily for MAGI-1
                temp_image_path = f"/tmp/magi1_init_frame_{sample_idx}.png"
                if isinstance(init_frame, Image.Image):
                    init_frame.save(temp_image_path)
                else:
                    Image.fromarray(init_frame).save(temp_image_path)

                # Temporary output path for this candidate
                candidate_video_path = video_path.replace('.mp4', f'_candidate_{sample_idx}.mp4')

                # Use MAGI-1's I2V (no guidance for rejection sampling baseline)
                pipe.run_image_to_video(
                    prompt=prompt,
                    image_path=temp_image_path,
                    output_path=candidate_video_path
                )

                # Load the generated video frames
                candidate_result = load_video(candidate_video_path)
                candidate_frames.append(candidate_result)

                # Compute V-JEPA loss for this candidate
                if encoder is not None:
                    # Convert frames to tensor for loss computation
                    # frames is a list of PIL Images, convert to tensor
                    frames_tensor = torch.stack([torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in candidate_result])
                    frames_tensor = frames_tensor.unsqueeze(0).float()  # Add batch dimension: [T, C, H, W] -> [1, T, C, H, W]
                    frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W] -> [1, C, T, H, W]

                    # Normalize to [-1, 1] range (assuming frames are in [0, 255])
                    frames_tensor = (frames_tensor / 127.5) - 1.0

                    with torch.no_grad():
                        context_frames, stride, window_size = _resolve_sliding_window_params(args)

                        loss = compute_vjepa_loss_sliding_window(
                            video_tensor=frames_tensor,
                            encoder=encoder,
                            target_encoder=target_encoder,
                            predictor=predictor,
                            img_size=getattr(args, 'vjepa_img_size', 256),
                            window_size=window_size,
                            loss_exp=2,
                            masking_mode=getattr(args, 'vjepa_masking_mode', 'causal'),
                            context_frames=context_frames,
                            mask_ratio=getattr(args, 'vjepa_mask_ratio', 0.75),
                            spatial_pred_mask_scale=None,
                            temporal_pred_mask_scale=None,
                            aspect_ratio=None,
                            npred=None,
                            max_context_frames_ratio=None,
                            is_vae_output=True,
                            seed=args.seed,
                            stride=stride,
                            mode=getattr(args, 'loss_mode', 'max')
                        )
                        candidate_losses.append(loss.item())
                        print(f"      Candidate {sample_idx + 1} V-JEPA loss: {loss.item():.6f}")

                # Clean up temporary candidate file
                if os.path.exists(candidate_video_path):
                    os.remove(candidate_video_path)

                else:
                    print(f"      Warning: V-JEPA models not loaded, using random selection")
                    candidate_losses.append(sample_idx)  # Use index as dummy loss

            # Select best candidate based on lowest loss
            best_idx = np.argmin(candidate_losses)
            frames = candidate_frames[best_idx]
            best_loss = candidate_losses[best_idx]
            print(f"    Selected candidate {best_idx + 1} with lowest V-JEPA loss: {best_loss:.6f}")

        # Export to video
        export_to_video(frames, video_path, fps=fps)
        if args.sampling_method == 'rejection':
            print(f"[{experiment_name}] Generated: {video_path} (selected from {args.rejection_samples} candidates)")
        else:
            print(f"[{experiment_name}] Generated: {video_path}")

    # Log experiment completion
    log_experiment_simple(args, experiment_name, 'completed')
    if getattr(args, 'output_path', None) and len(prompts) == 1:
        print(f"[{experiment_name}] Experiment completed! Saved: {os.path.abspath(args.output_path)}")
    else:
        print(f"[{experiment_name}] Experiment completed! Results saved to: {experiment_folder}")

def resolve_paths(input_video, input_image, output_video, base_dir):
    """Resolve input/output paths for Physics-IQ dataset."""
    # Physics-IQ: Use absolute paths, ignore base_dir for inputs
    input_video_abs = os.path.join("PhysicsIQ/code/physics-IQ-benchmark", input_video) if input_video else None
    input_image_abs = os.path.join("PhysicsIQ/code/physics-IQ-benchmark", input_image) if input_image else None
    # Output can still be relative to base_dir
    output_video_abs = output_video

    return input_video_abs, input_image_abs, output_video_abs

def chunk_prompts(prompts, num_chunks, chunk_idx):
    """Divide the prompts into chunks and return the chunk corresponding to the given index."""
    chunk_size = math.ceil(len(prompts) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(prompts))
    return prompts[start_idx:end_idx]

def main():
    parser = argparse.ArgumentParser(description="Generate videos from text prompts using MAGI-1 I2V.")
    parser.add_argument('--prompt_file', type=str, required=False, help='Path to the text file containing prompts.')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt text; overrides prompt_file when set.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to MAGI-1 configuration JSON file.')
    parser.add_argument('--output_folder', type=str, required=False, default="generated_videos", help='Folder to save the generated videos. If --output_path is set, its directory will be used instead.')
    parser.add_argument('--output_path', type=str, default=None, help='Explicit output video path (mp4) when using a single prompt.')
    parser.add_argument('--output_filename', type=str, default=None, help='Output filename (e.g., name.mp4) to use under the experiment folder when using a single prompt.')
    parser.add_argument('--config_version', type=str, default='v2', help='Configuration version tag for experiment naming and tracking.')
    parser.add_argument('--batch_json', type=str, default=None, help='Optional: JSON file with list of {input_video|input_image, prompt, output_video} entries to process. Entries will be sharded across GPUs by index modulo num_gpus.')
    parser.add_argument('--base_dir', type=str, default=None, help='Optional: Base directory to prepend to input/output paths in --batch_json.')
    parser.add_argument('--dataset_mode', type=str, default='physics_iq', choices=['physics_iq'],
                       help='Dataset mode for path resolution.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs available across all nodes.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='Global index of the GPU to use for this process (0 to num_gpus-1).')
    parser.add_argument('--num_nodes', type=int, default=1, help='Total number of nodes available.')
    parser.add_argument('--node_id', type=int, default=0, help='Index of the current node (0 to num_nodes-1).')
    parser.add_argument('--gpus_per_node', type=int, default=8, help='Number of GPUs per node.')
    parser.add_argument('--sampling_method', type=str, default='vanilla',
                       choices=['vanilla', 'guidance', 'rejection'],
                       help='Sampling method to use.')
    parser.add_argument('--init_image', type=str, default=None, help='Path to the initial image for I2V conditioning.')
    parser.add_argument('--init_video', type=str, default=None, help='Path to the initial video for I2V conditioning (first frame used).')

    # Generation parameters
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps.')
    parser.add_argument('--num_frames', type=int, default=49, help='Number of frames to generate.')
    parser.add_argument('--height', type=int, default=480, help='Height of the generated videos.')
    parser.add_argument('--width', type=int, default=720, help='Width of the generated videos.')
    parser.add_argument('--cfg_scale', type=float, default=6.0, help='Classifier-free guidance scale.')

    # Guidance sampling parameters
    parser.add_argument('--guidance_scale', type=float, default=0.001, help='VJEPA guidance scale.')
    parser.add_argument('--guidance_start', type=int, default=0, help='Legacy option kept for compatibility; not consumed by the current MAGI guidance path.')
    parser.add_argument('--guidance_end', type=int, default=1001, help='Legacy option kept for compatibility; not consumed by the current MAGI guidance path.')
    parser.add_argument('--guidance_rho_scale', type=float, default=6.0, help='Legacy option kept for compatibility; not consumed by the current MAGI guidance path.')
    parser.add_argument('--guidance_frequency', type=int, default=5, help='Frequency of guidance updates.')
    parser.add_argument('--travel_time', type=str, default='3,12', help='Legacy option kept for compatibility; not consumed by the current MAGI guidance path.')

    # VJEPA guidance parameters
    parser.add_argument('--guidance_step_pattern', type=str, default='0x3,3x12,2x12,1x23', help='Legacy option kept for compatibility; not consumed by the current MAGI guidance path.')
    parser.add_argument('--guidance_lr_pattern', type=str, default='3.0x15,2.0x15,1.0x20', help='Legacy option kept for compatibility; not consumed by the current MAGI guidance path.')
    parser.add_argument('--vjepa_variant', type=str, default='vit_giant', choices=['vith', 'vit_huge', 'vitg', 'vit_giant', 'vitg384', 'vit_giant_384', 'vitgac', 'vit_giant_ac'])
    parser.add_argument('--vjepa_type', type=str, default=None, choices=['vith', 'vit_huge', 'vitg', 'vit_giant', 'vitg384', 'vit_giant_384', 'vitgac', 'vit_giant_ac'], help='Optional override for the MAGI guidance JEPA backbone.')
    parser.add_argument('--vjepa_img_size', type=int, default=256)
    parser.add_argument('--style_weight', type=float, default=1.0)
    parser.add_argument('--vjepa_masking_mode', type=str, default='causal', choices=['causal', 'random'])
    parser.add_argument('--vjepa_context_frames', type=int, default=8, help='Used by rejection sampling and metadata only; not consumed by the current MAGI guidance path.')
    parser.add_argument('--vjepa_mask_ratio', type=float, default=0.5)
    parser.add_argument('--slice_window_size', type=int, default=16, help='Used by rejection sampling and metadata only; not consumed by the current MAGI guidance path.')
    parser.add_argument('--slice_stride', type=int, default=4, help='Used by rejection sampling and metadata only; not consumed by the current MAGI guidance path.')
    parser.add_argument('--vae_decode_scale', type=float, default=0.7, help='VAE decode scale factor.')
    parser.add_argument('--loss_mode', type=str, default='max', choices=['mean', 'max'], help='Used by rejection sampling and metadata only; not consumed by the current MAGI guidance path.')

    # Rejection sampling parameters (only used when sampling_method='rejection')
    parser.add_argument('--rejection_samples', type=int, default=3,
                       help='Number of samples to generate for rejection sampling.')

    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility.')
    args = parser.parse_args()

    # Set deterministic behavior for reproducibility
    set_deterministic(seed=args.seed)
    args.fps = 8

    # Generate simple experiment name
    experiment_name = get_simple_experiment_name(args)

    # Print configuration for this run
    print(f"\n{'='*60}")
    print(f"MAGI-1 EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Config: {args.config_file}")
    print(f"Sampling method: {args.sampling_method}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Frames per video: {args.num_frames}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Resolution: {args.height}x{args.width}")

    print(f"Running in batch_json mode. Multi-node setup:")
    print(f"  - Node {args.node_id + 1}/{args.num_nodes}")
    print(f"  - Local GPU {args.gpu_idx % args.gpus_per_node}, Global GPU {args.gpu_idx + 1}/{args.num_gpus}")
    print(f"  - Sharding entries by global GPU index {args.gpu_idx}")

    if args.sampling_method == 'guidance':
        print(f"Guidance parameters:")
        print(f"  - Scale: {args.guidance_scale}")
        print(f"  - Frequency: {args.guidance_frequency}")
        print(f"  - V-JEPA backbone: {normalize_vjepa_variant(args.vjepa_type or args.vjepa_variant)}")
        print("  - Note: legacy guidance tuning flags are metadata-only today and are not consumed by the current MAGI guidance path.")

    if args.sampling_method == 'rejection':
        print(f"Rejection sampling enabled:")
        print(f"  - Number of samples: {args.rejection_samples}")
        print(f"  - Using V-JEPA parameters for evaluation")

    print(f"{'='*60}\n")

    # Initialize pipeline once per process
    pipe = init_pipeline(args)

    # Initialize V-JEPA models for rejection sampling if enabled
    vjepa_models = init_vjepa_models(args)


    # Batch JSON mode: load tasks, shard by index, and process this shard
    with open(args.batch_json, 'r') as f:
        entries = json.load(f)

    base_dir = args.base_dir
    print("Using Physics-IQ mode: absolute input paths, relative output paths")

    # Use same chunking mechanism as vanilla/guidance methods for consistent ordering
    chunked_entries = chunk_prompts(entries, args.num_gpus, args.gpu_idx)

    negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"

    processed_count = 0
    for item in chunked_entries:

        input_video = item.get('input_video')
        input_image = item.get('input_image')
        prompt = item.get('prompt')
        output_video = item.get('output_video')
        if prompt is None or output_video is None or (input_video is None and input_image is None):
            print(f"[skip] Missing required fields in entry: {item}")
            continue

        # Resolve paths
        input_video_abs, input_image_abs, output_video = resolve_paths(
            input_video, input_image, output_video, base_dir
        )

        # Prepare per-item init frame
        if "5frame" not in args.batch_json:
            init_frame = load_first_frame(input_image_abs, input_video_abs)
        else:
            init_frame = load_video(input_video_abs)

        # Prepare prompts list and negative prompt
        per_item_prompts = [prompt]


        # Save under the configured output folder and experiment name.
        # Use the JSON-provided output filename, but place it inside
        # <output_folder>/<experiment>/ to match project structure.
        args.output_path = None
        args.output_filename = os.path.basename(output_video)

        args.init_image = input_image_abs
        args.init_video = input_video_abs


        print(f"[GPU {args.gpu_idx}] {os.path.basename(output_video)}")
        generate_videos(pipe, args, init_frame, per_item_prompts, negative_prompt, experiment_name, fps=args.fps, vjepa_models=vjepa_models)
        processed_count += 1

        print(f"Node {args.node_id} processed {processed_count} entries on global GPU index {args.gpu_idx}.")

if __name__ == "__main__":
    main()
