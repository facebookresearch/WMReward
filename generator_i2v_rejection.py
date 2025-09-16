import argparse
import json
from diffusers.utils import export_to_video, load_video
from datetime import datetime
import torch
import cv2
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import math
import numpy as np
import re
import shutil
from PIL import Image
from utils import compute_vjepa_loss_sliding_window, load_vjepa_models_torchhub

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

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

def load_grid_config(config_path="grid_config.json"):
    """Load grid search configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {"current_version": "v1", "experiment_versions": {}}

def save_experiment_metadata(args, experiment_name, experiment_folder):
    """Save experiment metadata as JSON file in the experiment folder."""
    
    # Create metadata with all relevant parameters
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config_version": getattr(args, 'config_version', 'v1'),
        "parameters": {
            "sampling_method": args.sampling_method,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "model_id": args.model_id,
            "prompt_file": getattr(args, 'prompt_file', None),
            "batch_json": getattr(args, 'batch_json', None),
            "height": getattr(args, 'height', 480),
            "width": getattr(args, 'width', 720),
        }
    }
    
    # Add guidance-specific parameters
    if args.sampling_method == 'guidance':
        context_frames, stride, window_size = _resolve_sliding_window_params(args)
        metadata["parameters"].update({
            "vjepa_context_frames": context_frames,
            "slice_stride": stride,
            "slice_window_size": window_size,
            "travel_time": getattr(args, 'travel_time', None),
            "guidance_step_pattern": getattr(args, 'guidance_step_pattern', None),
            "guidance_lr_pattern": getattr(args, 'guidance_lr_pattern', None),
            "guidance_frequency": getattr(args, 'guidance_frequency', None),
            "vjepa_variant": getattr(args, 'vjepa_variant', None),
            "vjepa_img_size": getattr(args, 'vjepa_img_size', None),
            "vjepa_masking_mode": getattr(args, 'vjepa_masking_mode', None),
            "vjepa_mask_ratio": getattr(args, 'vjepa_mask_ratio', None),
            "style_weight": getattr(args, 'style_weight', None),
            "loss_mode": getattr(args, 'loss_mode', None),
        })
    
    # Add rejection sampling parameters
    if args.sampling_method == 'rejection':
        metadata["parameters"].update({
            "rejection_samples": getattr(args, 'rejection_samples', 3),
            "loss_mode": getattr(args, 'loss_mode', 'mean'),
        })
    
    # Add rej_guide parameters (combines guidance and rejection)
    if args.sampling_method == 'rej_guide':
        context_frames, stride, window_size = _resolve_sliding_window_params(args)
        metadata["parameters"].update({
            "vjepa_context_frames": context_frames,
            "slice_stride": stride,
            "slice_window_size": window_size,
            "travel_time": getattr(args, 'travel_time', None),
            "guidance_step_pattern": getattr(args, 'guidance_step_pattern', None),
            "guidance_lr_pattern": getattr(args, 'guidance_lr_pattern', None),
            "guidance_frequency": getattr(args, 'guidance_frequency', None),
            "vjepa_variant": getattr(args, 'vjepa_variant', None),
            "vjepa_img_size": getattr(args, 'vjepa_img_size', None),
            "vjepa_masking_mode": getattr(args, 'vjepa_masking_mode', None),
            "vjepa_mask_ratio": getattr(args, 'vjepa_mask_ratio', None),
            "style_weight": getattr(args, 'style_weight', None),
            "loss_mode": getattr(args, 'loss_mode', None),
            "rejection_samples": getattr(args, 'rejection_samples', 3),
        })
    
    # Save metadata file
    metadata_path = os.path.join(experiment_folder, "experiment_config.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved experiment metadata to: {metadata_path}")

def get_simple_experiment_name(args):
    """Generate a clean, short experiment folder name (with optional version tag)."""

    version = getattr(args, 'config_version', 'v1')

    # Base name with method and key params
    if args.sampling_method == 'vanilla':
        name = f"vanilla_{version}_f{args.num_frames}_s{args.num_inference_steps}_cfg{args.cfg_scale}"
    elif args.sampling_method == 'guidance':
        loss_mode = getattr(args, 'loss_mode', 'mean')
        vjepa_variant = getattr(args, 'vjepa_variant', 'vit_giant')
        # Simplify vjepa variant name for brevity
        vjepa_short = vjepa_variant.replace('vit_', '') if vjepa_variant != 'vit_giant' else ''
        
        name = (
            f"guidance_{version}_f{args.num_frames}_s{args.num_inference_steps}"
            f"_c{getattr(args, 'vjepa_context_frames', 8)}"
            f"_cfg{args.cfg_scale}_{loss_mode}"
        )
        
        # Add vjepa variant if not default
        if vjepa_short:
            name += f"_{vjepa_short}"
    elif args.sampling_method == 'rejection':
        name = f"{args.sampling_method}_{version}_f{args.num_frames}_s{args.num_inference_steps}_cfg{args.cfg_scale}"
        name += f"_reject{getattr(args, 'rejection_samples', 3)}_{getattr(args, 'loss_mode', 'mean')}"
        vjepa_variant = getattr(args, 'vjepa_variant', 'vit_giant')

        name += f"_{vjepa_variant}_abl"
    elif args.sampling_method == 'rej_guide':
        loss_mode = getattr(args, 'loss_mode', 'mean')
        vjepa_variant = getattr(args, 'vjepa_variant', 'vit_giant')
        # Simplify vjepa variant name for brevity
        vjepa_short = vjepa_variant.replace('vit_', '') if vjepa_variant != 'vit_giant' else ''
        
        name = (
            f"rej_guide_{version}_f{args.num_frames}_s{args.num_inference_steps}"
            f"_c{getattr(args, 'vjepa_context_frames', 8)}"
            f"_cfg{args.cfg_scale}_reject{getattr(args, 'rejection_samples', 3)}_{loss_mode}"
        )
        
        # Add vjepa variant if not default
        if vjepa_short:
            name += f"_{vjepa_short}"

    if "5frame" in args.batch_json:
        name += "_5frame"

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
    """Initialize the CogVideoX I2V pipeline with V-JEPA slice-pred support."""
    if "CogVideoX" in args.model_id:
        from pipelines.pipeline_cogvideox_image2video_vjepa import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to("cuda")
        pipe.vae.enable_tiling(); pipe.vae.enable_slicing(); pipe.vae.enable_gradient_checkpointing()
    elif "Cosmos" in args.model_id:
        # from pipelines.pipeline_cosmos_image2video_v1 import Cosmos2VideoToWorldPipeline
        # from pipelines.pipeline_cosmos_image2video_14b import Cosmos2VideoToWorldPipeline
        from pipelines.pipeline_cosmos_image2video_v2 import Cosmos2VideoToWorldPipeline
        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to("cuda")
        pipe = pipe.to("cuda")
        pipe.transformer.enable_gradient_checkpointing()

        # pipe.enable_sequential_cpu_offload() # it needs to be enabled for the model to run on A6000
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe

def init_vjepa_models(args):
    """Initialize V-JEPA models for rejection sampling evaluation."""
    if args.sampling_method not in ['rejection', 'rej_guide']:
        return None, None, None
    
    print(f"Loading V-JEPA models for {args.sampling_method} sampling ({args.vjepa_variant})...")
    encoder, target_encoder, predictor, img_size = load_vjepa_models_torchhub(args.vjepa_variant)
    encoder.eval().cuda()
    target_encoder.eval().cuda()
    predictor.eval().cuda()
    print(f"V-JEPA models loaded successfully (img_size: {img_size})")
    return encoder, target_encoder, predictor

## Legacy rejection sampler removed

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


def guidance_sample(pipe, args, init_frame, prompt, negative_prompt, generator=None):
    """Guidance sampling using slice-predictor loss for I2V pipeline."""
    
    steps = int(args.num_inference_steps)
    step_pattern = getattr(args, 'guidance_step_pattern', "0x3,3x12,2x12,1x23")
    lr_pattern = getattr(args, 'guidance_lr_pattern', "3.0x15,2.0x15,1.0x20")

    guidance_step = _build_seq(step_pattern, steps, is_float=False)
    guidance_lr = _build_seq(lr_pattern, steps, is_float=True)
    travel_time = parse_range_pair(args.travel_time)

    # Unify sliding-window params
    context_frames, stride, window_size = _resolve_sliding_window_params(args)

    if "CogVideoX" in args.model_id:
        result = pipe(
            video=[init_frame],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=args.cfg_scale,
            use_dynamic_cfg=True,
            fixed_frames=None,
            guidance_step=guidance_step,
            guidance_lr=guidance_lr,
            guidance_frequency=args.guidance_frequency,
            additional_inputs={
                "vjepa_variant": getattr(args, 'vjepa_variant', "vit_giant"),
                "vjepa_img_size": int(getattr(args, 'vjepa_img_size', 256)),
                "vjepa_masking_mode": str(getattr(args, 'vjepa_masking_mode', 'causal')),
                "vjepa_context_frames": context_frames,
                "vjepa_mask_ratio": float(getattr(args, 'vjepa_mask_ratio', 0.75)),
                "slice_window_size": window_size,
                "slice_stride": stride,
                "vae_decode_scale": float(getattr(args, 'vae_decode_scale', 0.7)),
                "loss_mode": getattr(args, 'loss_mode', 'max'),
            },
            travel_time=travel_time,
        )
    elif "Cosmos" in args.model_id:
        additional_inputs = {
                "vjepa_variant": getattr(args, 'vjepa_variant', "vit_giant"),
                "vjepa_img_size": int(getattr(args, 'vjepa_img_size', 256)),
                "vjepa_masking_mode": str(getattr(args, 'vjepa_masking_mode', 'causal')),
                "vjepa_context_frames": context_frames,
                "vjepa_mask_ratio": float(getattr(args, 'vjepa_mask_ratio', 0.75)),
                "slice_window_size": window_size,
                "slice_stride": stride,
                "vae_decode_scale": float(getattr(args, 'vae_decode_scale', 0.7)),
                "loss_mode": getattr(args, 'loss_mode', 'max'),
                "save_intermediate": False,  # Enable/disable intermediate saving
            }
        if "5frame" in args.batch_json:
            result = pipe(
                video=init_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=args.cfg_scale,
                generator=generator,
                guidance_step=guidance_step,
                guidance_lr=guidance_lr,
                guidance_frequency=args.guidance_frequency,
                additional_inputs=additional_inputs,
                fps=16,
                travel_time=travel_time,
            )

        else:
            result = pipe(
                image=init_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=args.cfg_scale,
                generator=generator,
                guidance_step=guidance_step,
                guidance_lr=guidance_lr,
                guidance_frequency=args.guidance_frequency,
                additional_inputs=additional_inputs,
                fps=16,
                travel_time=travel_time,
            )
        
    return result.frames[0]

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
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            # Use I2V pipeline in a no-guidance mode by passing zero repeats
            zero_steps = [0] * int(args.num_inference_steps)
            zero_lrs = [0.0] * int(args.num_inference_steps)
            if "CogVideoX" in args.model_id:
                result = pipe(
                    video=[init_frame],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    use_dynamic_cfg=True,
                    fixed_frames=None,
                    guidance_step=zero_steps,
                    guidance_lr=zero_lrs,
                    travel_time=(0, 0),
                    additional_inputs={
                        "loss_mode": args.loss_mode,
                    },
                )
            elif "Cosmos" in args.model_id:
                if "5frame" in args.batch_json:
                    result = pipe(
                        video=init_frame,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.cfg_scale,
                        generator=generator,
                        guidance_step=zero_steps,
                        guidance_lr=zero_lrs,
                        guidance_frequency=args.guidance_frequency,
                        additional_inputs={
                            "loss_mode": args.loss_mode,
                        },
                        travel_time=(0, 0),
                    )
                else:
                    result = pipe(
                        image=init_frame,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.cfg_scale,
                        generator=generator,
                        guidance_step=zero_steps,
                        guidance_lr=zero_lrs,
                        guidance_frequency=args.guidance_frequency,
                        additional_inputs={
                            "loss_mode": args.loss_mode,
                        },
                        travel_time=(0, 0),
                    )
            frames = result.frames[0]
        elif args.sampling_method == 'guidance':
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            frames = guidance_sample(
                pipe=pipe,
                args=args,
                init_frame=init_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator
            )
          

        elif args.sampling_method == 'rejection':
            print(f"  Generating with rejection sampling ({args.rejection_samples} candidates)...")
            
            # Use centralized buffer folder only
            base_experiment_folder = os.path.dirname(experiment_folder)
            buffer_experiment_pattern = f"rejection_{getattr(args, 'vjepa_variant', 'vit_giant')}_{args.loss_mode}_buffer"
            buffer_experiment_folder = os.path.join(base_experiment_folder, buffer_experiment_pattern)
            
            # Define buffer subfolder for this prompt
            if "physics" in args.batch_json:
                buffer_rejection_folder = os.path.join(buffer_experiment_folder, f"{args.output_filename}_rejection")
            else:
                buffer_rejection_folder = os.path.join(buffer_experiment_folder, f"{safe_prompt}_rejection")
            
            print(f"Using buffer folder: {buffer_rejection_folder}")
            
            # Check for existing candidates in buffer
            existing_candidates = []
            if os.path.exists(buffer_rejection_folder):
                print(f"    Found existing rejection buffer, checking for candidates...")
                
                # Load candidates from buffer (check more than needed in case some are missing)
                for i in range(16):  # Check up to 32 candidates
                    candidate_path = os.path.join(buffer_rejection_folder, f"candidate_{i+1:02d}.mp4")
                    loss_path = os.path.join(buffer_rejection_folder, f"candidate_{i+1:02d}_loss.txt")
                    if os.path.exists(candidate_path) and os.path.exists(loss_path):
                        with open(loss_path, 'r') as f:
                            loss = float(f.read().strip())
                        existing_candidates.append((i+1, candidate_path, loss))
            
            if len(existing_candidates) >= args.rejection_samples:
                print(f"    Found {len(existing_candidates)} existing candidates, sampling {args.rejection_samples} from buffer...")
                
                # Sample k candidates from the buffer (with or without replacement depending on buffer size)
                if len(existing_candidates) == args.rejection_samples:
                    # Use all available candidates
                    candidates_to_use = existing_candidates
                    print(f"      Using all {len(existing_candidates)} available candidates")
                else:
                    # Sample k candidates from the buffer
                    candidate_indices = np.random.choice(len(existing_candidates), args.rejection_samples, replace=False)
                    candidates_to_use = [existing_candidates[i] for i in candidate_indices]
                    print(f"      Sampled {args.rejection_samples} candidates from {len(existing_candidates)} available")
                
                # Load the sampled candidates and their losses
                candidate_frames = []
                candidate_losses = []
                for idx, candidate_path, loss in candidates_to_use:
                    print(f"      Loading candidate {idx} with loss: {loss:.6f}")
                    video_frames = load_video(candidate_path)
                    candidate_frames.append(video_frames)
                    candidate_losses.append(loss)
            else:
                print(f"    Found {len(existing_candidates)} existing candidates, generating {args.rejection_samples - len(existing_candidates)} more in buffer...")
                
                # Create buffer folder if it doesn't exist
                os.makedirs(buffer_rejection_folder, exist_ok=True)
                
                candidate_frames = []
                candidate_losses = []
                
                # Load existing candidates first
                for idx, candidate_path, loss in existing_candidates:
                    print(f"      Loading existing candidate {idx} with loss: {loss:.6f}")
                    video_frames = load_video(candidate_path)
                    candidate_frames.append(video_frames)
                    candidate_losses.append(loss)
                
                # Generate missing candidates in buffer folder
                for sample_idx in range(len(existing_candidates), args.rejection_samples):
                    print(f"    Generating candidate {sample_idx + 1}/{args.rejection_samples}...")
                    
                    # Create unique generator for each sample
                    sample_generator = torch.Generator(device="cuda").manual_seed(args.seed + sample_idx)
                    
                    # Generate using vanilla method (no guidance)
                    zero_steps = [0] * int(args.num_inference_steps)
                    zero_lrs = [0.0] * int(args.num_inference_steps)
                    
                    if "CogVideoX" in args.model_id:
                        result = pipe(
                            video=[init_frame],
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_frames=args.num_frames,
                            height=args.height,
                            width=args.width,
                            generator=sample_generator,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.cfg_scale,
                            use_dynamic_cfg=True,
                            fixed_frames=None,
                            guidance_step=zero_steps,
                            guidance_lr=zero_lrs,
                            travel_time=(0, 0),
                            additional_inputs=None,
                        )
                    elif "Cosmos" in args.model_id:
                        if "5frame" in args.batch_json:
                            result = pipe(
                                video=init_frame,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.cfg_scale,
                                generator=sample_generator,
                                guidance_step=zero_steps,
                                guidance_lr=zero_lrs,
                                guidance_frequency=args.guidance_frequency,
                                additional_inputs=None,
                                travel_time=(0, 0),
                            )
                        else:
                            result = pipe(
                                image=init_frame,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.cfg_scale,
                                generator=sample_generator,
                                guidance_step=zero_steps,
                                guidance_lr=zero_lrs,
                                guidance_frequency=args.guidance_frequency,
                                additional_inputs=None,
                                travel_time=(0, 0),
                            )
                    
                    # Save candidate video in buffer folder
                    candidate_path = os.path.join(buffer_rejection_folder, f"candidate_{sample_idx+1:02d}.mp4")
                    export_to_video(result.frames[0], candidate_path, fps=args.fps)
                    candidate_frames.append(result.frames[0])
                    
                    # Compute V-JEPA loss for this candidate
                    if encoder is not None:
                        # Convert frames to tensor for loss computation
                        frames_tensor = torch.stack([torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in result.frames[0]])
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
                                mode=getattr(args, 'loss_mode', 'mean')
                            )
                            candidate_losses.append(loss.item())
                            print(f"      Candidate {sample_idx + 1} V-JEPA loss: {loss.item():.6f}")
                            
                            # Save loss to file in buffer folder
                            loss_path = os.path.join(buffer_rejection_folder, f"candidate_{sample_idx+1:02d}_loss.txt")
                            with open(loss_path, 'w') as f:
                                f.write(f"{loss.item():.6f}")

                    else:
                        print(f"      Warning: V-JEPA models not loaded, using random selection")
                        candidate_losses.append(sample_idx)  # Use index as dummy loss
                        
                        # Save dummy loss to file in buffer folder
                        loss_path = os.path.join(buffer_rejection_folder, f"candidate_{sample_idx+1:02d}_loss.txt")
                        with open(loss_path, 'w') as f:
                            f.write(f"{sample_idx}")
            
            # Select best candidate based on lowest loss (consistent for all cases)
            best_idx = np.argmin(candidate_losses)
            frames = candidate_frames[best_idx]
            best_loss = candidate_losses[best_idx]
            print(f"    Selected candidate {best_idx + 1} with lowest V-JEPA loss: {best_loss:.6f}")
            

        elif args.sampling_method == 'rej_guide':
            print(f"  Generating with rejection + guidance sampling ({args.rejection_samples} candidates)...")
            
            # Use centralized buffer folder only
            base_experiment_folder = os.path.dirname(experiment_folder)
            buffer_experiment_pattern = f"rejguide_{getattr(args, 'vjepa_variant', 'vit_giant')}_buffer"
            buffer_experiment_folder = os.path.join(base_experiment_folder, buffer_experiment_pattern)
            
            # Define buffer subfolder for this prompt
            if "physics" in args.batch_json:
                buffer_rejection_folder = os.path.join(buffer_experiment_folder, f"{args.output_filename}_rejection")
            else:
                buffer_rejection_folder = os.path.join(buffer_experiment_folder, f"{safe_prompt}_rejection")
            
            print(f"Using buffer folder: {buffer_rejection_folder}")
            
            # Check for existing candidates in buffer
            existing_candidates = []
            if os.path.exists(buffer_rejection_folder):
                print(f"    Found existing rejection buffer, checking for candidates...")
                
                # Load candidates from buffer (check more than needed in case some are missing)
                for i in range(16):  # Check up to 32 candidates
                    candidate_path = os.path.join(buffer_rejection_folder, f"candidate_{i+1:02d}.mp4")
                    loss_path = os.path.join(buffer_rejection_folder, f"candidate_{i+1:02d}_loss.txt")
                    if os.path.exists(candidate_path) and os.path.exists(loss_path):
                        with open(loss_path, 'r') as f:
                            loss = float(f.read().strip())
                        existing_candidates.append((i+1, candidate_path, loss))
            
            if len(existing_candidates) >= args.rejection_samples:
                print(f"    Found {len(existing_candidates)} existing candidates, sampling {args.rejection_samples} from buffer...")
                
                # Sample k candidates from the buffer (with or without replacement depending on buffer size)
                if len(existing_candidates) == args.rejection_samples:
                    # Use all available candidates
                    candidates_to_use = existing_candidates
                    print(f"      Using all {len(existing_candidates)} available candidates")
                else:
                    # Sample k candidates from the buffer
                    candidate_indices = np.random.choice(len(existing_candidates), args.rejection_samples, replace=False)
                    candidates_to_use = [existing_candidates[i] for i in candidate_indices]
                    print(f"      Sampled {args.rejection_samples} candidates from {len(existing_candidates)} available")
                
                # Load the sampled candidates and their losses
                candidate_frames = []
                candidate_losses = []
                for idx, candidate_path, loss in candidates_to_use:
                    print(f"      Loading candidate {idx} with loss: {loss:.6f}")
                    video_frames = load_video(candidate_path)
                    candidate_frames.append(video_frames)
                    candidate_losses.append(loss)
            else:
                print(f"    Found {len(existing_candidates)} existing candidates, generating {args.rejection_samples - len(existing_candidates)} more in buffer...")
                
                # Create buffer folder if it doesn't exist
                os.makedirs(buffer_rejection_folder, exist_ok=True)
                
                candidate_frames = []
                candidate_losses = []
                
                # Load existing candidates first
                for idx, candidate_path, loss in existing_candidates:
                    print(f"      Loading existing candidate {idx} with loss: {loss:.6f}")
                    video_frames = load_video(candidate_path)
                    candidate_frames.append(video_frames)
                    candidate_losses.append(loss)
                
                # Generate missing candidates using guidance sampling in buffer folder
                for sample_idx in range(len(existing_candidates), args.rejection_samples):
                    print(f"    Generating guided candidate {sample_idx + 1}/{args.rejection_samples}...")
                    
                    # Create unique generator for each sample
                    sample_generator = torch.Generator(device="cuda").manual_seed(args.seed + sample_idx)
                    
                    # Generate using guidance sampling
                    guided_frames = guidance_sample(
                        pipe=pipe,
                        args=args,
                        init_frame=init_frame,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        generator=sample_generator
                    )
                    
                    # Save candidate video in buffer folder
                    candidate_path = os.path.join(buffer_rejection_folder, f"candidate_{sample_idx+1:02d}.mp4")
                    export_to_video(guided_frames, candidate_path, fps=args.fps)
                    candidate_frames.append(guided_frames)
                    
                    # Compute V-JEPA loss for this candidate
                    if encoder is not None:
                        # Convert frames to tensor for loss computation
                        frames_tensor = torch.stack([torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in guided_frames])
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
                                mode=getattr(args, 'loss_mode', 'mean')
                            )
                            candidate_losses.append(loss.item())
                            print(f"      Candidate {sample_idx + 1} V-JEPA loss: {loss.item():.6f}")
                            
                            # Save loss to file in buffer folder
                            loss_path = os.path.join(buffer_rejection_folder, f"candidate_{sample_idx+1:02d}_loss.txt")
                            with open(loss_path, 'w') as f:
                                f.write(f"{loss.item():.6f}")
                    else:
                        print(f"      Warning: V-JEPA models not loaded, using random selection")
                        candidate_losses.append(sample_idx)  # Use index as dummy loss
                        
                        # Save dummy loss to file in buffer folder
                        loss_path = os.path.join(buffer_rejection_folder, f"candidate_{sample_idx+1:02d}_loss.txt")
                        with open(loss_path, 'w') as f:
                            f.write(f"{sample_idx}")
            
            # Select best candidate based on lowest loss
            best_idx = np.argmin(candidate_losses)
            frames = candidate_frames[best_idx]
            best_loss = candidate_losses[best_idx]
            print(f"    Selected candidate {best_idx + 1} with lowest V-JEPA loss: {best_loss:.6f}")
            

        # Export to video
        export_to_video(frames, video_path, fps=fps)
        if args.sampling_method in ['rejection', 'rej_guide']:
            method_name = "rejection + guidance" if args.sampling_method == 'rej_guide' else "rejection"
            print(f"[{experiment_name}] Generated: {video_path} (selected from {args.rejection_samples} {method_name} candidates)")
        else:
            print(f"[{experiment_name}] Generated: {video_path}")
    
    # Log experiment completion
    log_experiment_simple(args, experiment_name, 'completed')
    if getattr(args, 'output_path', None) and len(prompts) == 1:
        print(f"[{experiment_name}] Experiment completed! Saved: {os.path.abspath(args.output_path)}")
    else:
        print(f"[{experiment_name}] Experiment completed! Results saved to: {experiment_folder}")

def detect_dataset_mode(batch_json_path):
    """Auto-detect dataset mode from batch JSON filename."""
    if not batch_json_path:
        return 'dreambench'  # default
    
    filename = os.path.basename(batch_json_path).lower()
    if 'physics' in filename:
        return 'physics_iq'
    else:
        return 'dreambench'

def resolve_paths(input_video, input_image, output_video, base_dir, dataset_mode):
    """Resolve input/output paths based on dataset mode."""
    if dataset_mode == 'physics_iq':
        # Physics-IQ: Use absolute paths, ignore base_dir for inputs
        input_video_abs = os.path.join("/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark", input_video) if input_video else None
        input_image_abs = os.path.join("/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark", input_image) if input_image else None
        # Output can still be relative to base_dir
        output_video_abs = output_video
    else:
        pass
    
    return input_video_abs, input_image_abs, output_video_abs

def chunk_prompts(prompts, num_chunks, chunk_idx):
    """Divide the prompts into chunks and return the chunk corresponding to the given index."""
    chunk_size = math.ceil(len(prompts) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(prompts))
    return prompts[start_idx:end_idx]

def main():
    parser = argparse.ArgumentParser(description="Generate videos from text prompts using CogVideoX I2V.")
    parser.add_argument('--prompt_file', type=str, required=False, help='Path to the text file containing prompts.')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt text; overrides prompt_file when set.')
    parser.add_argument('--model_id', type=str, default="THUDM/CogVideoX-5b-I2V", help='CogVideoX I2V model ID to use for video generation.')
    parser.add_argument('--output_folder', type=str, required=False, default="generated_videos", help='Folder to save the generated videos. If --output_path is set, its directory will be used instead.')
    parser.add_argument('--output_path', type=str, default=None, help='Explicit output video path (mp4) when using a single prompt.')
    parser.add_argument('--output_filename', type=str, default=None, help='Output filename (e.g., name.mp4) to use under the experiment folder when using a single prompt.')
    parser.add_argument('--config_version', type=str, default='v1', help='Configuration version tag for experiment naming and tracking.')
    parser.add_argument('--batch_json', type=str, default=None, help='Optional: JSON file with list of {input_video|input_image, prompt, output_video} entries to process. Entries will be sharded across GPUs by index modulo num_gpus.')
    parser.add_argument('--base_dir', type=str, default=None, help='Optional: Base directory to prepend to input/output paths in --batch_json.')
    parser.add_argument('--dataset_mode', type=str, default='auto', choices=['auto', 'dreambench', 'physics_iq'], 
                       help='Dataset mode: auto (detect from batch_json name), dreambench (relative paths), physics_iq (absolute paths)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs available across all nodes.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='Global index of the GPU to use for this process (0 to num_gpus-1).')
    parser.add_argument('--num_nodes', type=int, default=1, help='Total number of nodes available.')
    parser.add_argument('--node_id', type=int, default=0, help='Index of the current node (0 to num_nodes-1).')
    parser.add_argument('--gpus_per_node', type=int, default=8, help='Number of GPUs per node.')
    parser.add_argument('--sampling_method', type=str, default='vanilla', 
                       choices=['vanilla', 'guidance', 'rejection', 'rej_guide'],
                       help='Sampling method to use.')
    parser.add_argument('--init_image', type=str, default=None, help='Path to the initial image for I2V conditioning.')
    parser.add_argument('--init_video', type=str, default=None, help='Path to the initial video for I2V conditioning (first frame used).')
    
    # CogVideoX specific parameters
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps.')
    parser.add_argument('--num_frames', type=int, default=49, help='Number of frames to generate (CogVideoX default: 49).')
    parser.add_argument('--height', type=int, default=480, help='Height of the generated videos.')
    parser.add_argument('--width', type=int, default=720, help='Width of the generated videos.')
    parser.add_argument('--cfg_scale', type=float, default=6.0, help='Classifier-free guidance scale.')
    
    # Guidance sampling parameters
    parser.add_argument('--guidance_start', type=int, default=0, help='Timestep to start applying guidance (0..1001).')
    parser.add_argument('--guidance_end', type=int, default=1001, help='Timestep to end applying guidance (0..1001).')
    parser.add_argument('--guidance_rho_scale', type=float, default=6.0, help='[Deprecated in slice_pred] Overall LR scale (use guidance_lr_pattern instead).')
    parser.add_argument('--guidance_frequency', type=int, default=5, help='Frequency of guidance updates.')
    parser.add_argument('--travel_time', type=str, default='3,12', help='Direct guidance range in global steps "start,end" (0..49). Overrides guidance_start/end if provided.')

    # Slice-pred specific (match run_vjepa_slicepred.py)
    parser.add_argument('--guidance_step_pattern', type=str, default='0x3,3x12,2x12,1x23', help='Repeat counts per step bucket.')
    parser.add_argument('--guidance_lr_pattern', type=str, default='3.0x15,2.0x15,1.0x20', help='LR per step bucket.')
    parser.add_argument('--vjepa_variant', type=str, default='vit_giant', choices=['vit_large','vit_huge','vit_giant','vit_giant_384'])
    parser.add_argument('--vjepa_img_size', type=int, default=256)
    parser.add_argument('--style_weight', type=float, default=1.0)
    parser.add_argument('--vjepa_masking_mode', type=str, default='causal', choices=['causal', 'random'])
    parser.add_argument('--vjepa_context_frames', type=int, default=8)
    parser.add_argument('--vjepa_mask_ratio', type=float, default=0.75)
    parser.add_argument('--slice_window_size', type=int, default=16)
    parser.add_argument('--slice_stride', type=int, default=4)
    parser.add_argument('--vae_decode_scale', type=float, default=0.7, help='VAE decode scale factor.')
    parser.add_argument('--loss_mode', type=str, default='max', choices=['mean', 'max'], help='V-JEPA loss aggregation mode.')
    
    # Rejection sampling parameters (only used when sampling_method='rejection')
    parser.add_argument('--rejection_samples', type=int, default=3, 
                       help='Number of samples to generate for rejection sampling.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility.')
    
    args = parser.parse_args()

    # Set deterministic behavior for reproducibility
    set_deterministic(seed=args.seed)
    if "Cosmos" in args.model_id:
        args.fps = 16
    elif "CogVideoX" in args.model_id:
        args.fps = 8
    else:
        args.fps = 8

    # Generate simple experiment name
    experiment_name = get_simple_experiment_name(args)

    # Print configuration for this run
    print(f"\n{'='*60}")
    print(f"COGVIDEOX EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Model: {args.model_id}")
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
        print(f"  - Step pattern: {args.guidance_step_pattern}")
        print(f"  - LR pattern: {args.guidance_lr_pattern}")
        print(f"  - Frequency: {args.guidance_frequency}")
        print(f"  - Travel time: {args.travel_time}")
    
    if args.sampling_method == 'rejection':
        print(f"Rejection sampling enabled:")
        print(f"  - Number of samples: {args.rejection_samples}")
        print(f"  - Using existing V-JEPA parameters for evaluation")
    
    if args.sampling_method == 'rej_guide':
        print(f"Rejection + Guidance sampling enabled:")
        print(f"  - Number of samples: {args.rejection_samples}")
        print(f"  - Step pattern: {args.guidance_step_pattern}")
        print(f"  - LR pattern: {args.guidance_lr_pattern}")
        print(f"  - Frequency: {args.guidance_frequency}")
        print(f"  - Travel time: {args.travel_time}")
        print(f"  - Using V-JEPA parameters for evaluation")
    
    print(f"{'='*60}\n")

    # Initialize pipeline once per process
    pipe = init_pipeline(args)
    
    # Initialize V-JEPA models for rejection sampling if enabled
    vjepa_models = init_vjepa_models(args)


    # Batch JSON mode: load tasks, shard by index, and process this shard
    with open(args.batch_json, 'r') as f:
        entries = json.load(f)
    # Determine dataset mode and base directory
    dataset_mode = args.dataset_mode if args.dataset_mode != 'auto' else detect_dataset_mode(args.batch_json)
    base_dir = args.base_dir
    
    print(f"Dataset mode: {dataset_mode}")
    if dataset_mode == 'physics_iq':
        print("Using Physics-IQ mode: absolute input paths, relative output paths")
    else:
        print("Using DreamBench mode: relative paths with base_dir")
    
    # Use same chunking mechanism as vanilla/guidance methods for consistent ordering
    chunked_entries = chunk_prompts(entries, args.num_gpus, args.gpu_idx)
    if "CogVideoX" in args.model_id:
        negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"
    elif "Cosmos" in args.model_id:
        negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
    
    processed_count = 0
    for item in chunked_entries:
        input_video = item.get('input_video')
        input_image = item.get('input_image')
        prompt = item.get('prompt')
        output_video = item.get('output_video')
        if prompt is None or output_video is None or (input_video is None and input_image is None):
            print(f"[skip] Missing required fields in entry: {item}")
            continue

        # Resolve paths based on dataset mode
        input_video_abs, input_image_abs, output_video = resolve_paths(
            input_video, input_image, output_video, base_dir, dataset_mode
        )

        # Prepare per-item init frame
        if "5frame" not in args.batch_json:
            init_frame = load_first_frame(input_image_abs, input_video_abs)
        else:
            init_frame = load_video(input_video_abs)

        # Prepare prompts list and negative prompt
        per_item_prompts = [prompt]

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
