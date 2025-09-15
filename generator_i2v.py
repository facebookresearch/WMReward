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

    # Add rejection samples suffix if using rejection sampling
    if args.sampling_method == 'rejection':
        name += f"_reject{getattr(args, 'rejection_samples', 3)}_{getattr(args, 'loss_mode', 'mean')}"

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

def get_prompts(prompt_file, args):
    """Read prompts and negative prompts from a text file."""
    with open(f"./prompts/{prompt_file}.txt", 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    # Define a negative prompt suitable for CogVideoX
    if "CogVideoX" in args.model_id:
        negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"
    elif "Cosmos" in args.model_id:
        negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
    
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
    """Initialize the CogVideoX I2V pipeline with V-JEPA slice-pred support."""
    if "CogVideoX" in args.model_id:
        from pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to("cuda")
        pipe.vae.enable_tiling(); pipe.vae.enable_slicing(); pipe.vae.enable_gradient_checkpointing()
    elif "Cosmos" in args.model_id:
        from pipelines.pipeline_cosmos_image2video import Cosmos2VideoToWorldPipeline
        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to("cuda")
        pipe = pipe.to("cuda")
        pipe.transformer.enable_gradient_checkpointing()

        # pipe.enable_sequential_cpu_offload() # it needs to be enabled for the model to run on A6000
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe

def init_vjepa_models(args):
    """Initialize V-JEPA models for rejection sampling evaluation."""
    if args.sampling_method != 'rejection':
        return None, None, None
    
    print(f"Loading V-JEPA models for rejection sampling ({args.vjepa_variant})...")
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
            loss_fn="slice_pred",
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
                loss_fn="slice_pred",
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
                loss_fn="slice_pred",
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

    # generator = torch.Generator(device="cuda").manual_seed(42)
    
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
            generator = torch.Generator(device="cuda").manual_seed(42)
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
                    loss_fn="slice_pred",
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
                        loss_fn="slice_pred",
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
                        loss_fn="slice_pred",
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
            generator = torch.Generator(device="cuda").manual_seed(42)
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
            
            # Generate multiple samples and select best based on V-JEPA loss
            candidate_frames = []
            candidate_losses = []
            
            for sample_idx in range(args.rejection_samples):
                print(f"    Generating candidate {sample_idx + 1}/{args.rejection_samples}...")
                
                # Create unique generator for each sample
                sample_generator = torch.Generator(device="cuda").manual_seed(42 + sample_idx)
                
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
                        loss_fn="slice_pred",
                        travel_time=(0, 0),
                        additional_inputs={
                            "loss_mode": getattr(args, 'loss_mode', 'mean'),
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
                            generator=sample_generator,
                            loss_fn="slice_pred",
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
                            generator=sample_generator,
                            loss_fn="slice_pred",
                            guidance_step=zero_steps,
                            guidance_lr=zero_lrs,
                            guidance_frequency=args.guidance_frequency,
                            additional_inputs={
                                "loss_mode": getattr(args, 'loss_mode', 'mean'),
                            },
                            travel_time=(0, 0),
                        )
                candidate_frames.append(result.frames[0])
                
                # Compute V-JEPA loss for this candidate
                if encoder is not None:
                    # Convert frames to tensor for loss computation
                    # frames is a list of PIL Images, convert to tensor
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
                            seed=42,
                            stride=stride,
                            mode=getattr(args, 'loss_mode', 'mean')
                        )
                        candidate_losses.append(loss.item())
                        print(f"      Candidate {sample_idx + 1} V-JEPA loss: {loss.item():.6f}")

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
        input_video_abs = input_video if input_video else None
        input_image_abs = input_image if input_image else None
        # Output can still be relative to base_dir
        output_video_abs = output_video if os.path.isabs(output_video) else os.path.join(base_dir, output_video)
    else:
        # DreamBench: Use relative paths with base_dir
        input_video_abs = os.path.join(base_dir, input_video) if input_video else None
        input_image_abs = os.path.join(base_dir, input_image) if input_image else None
        output_video_abs = os.path.join(base_dir, output_video)
    
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
    parser.add_argument('--num_gpus', type=int, default=1, help='Total number of GPUs available.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='Index of the GPU to use for this process.')
    parser.add_argument('--sampling_method', type=str, default='vanilla', 
                       choices=['vanilla', 'guidance', 'rejection'],
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
    
    args = parser.parse_args()

    # Set deterministic behavior for reproducibility
    # set_deterministic(seed=42)
    # print("Deterministic mode enabled (seed=42)")
    if "Cosmos" in args.model_id:
        args.fps = 16
    elif "CogVideoX" in args.model_id:
        args.fps = 8
    else:
        args.fps = 8

    # Validation
    if args.sampling_method == 'vanilla_lora' and not args.lora_path:
        raise ValueError("lora_path must be specified when using vanilla_lora sampling method")
    
    # In batch mode, per-item inputs are provided from JSON. Otherwise, require one of init_image/init_video
    if not args.batch_json:
        if not args.init_image and not args.init_video:
            raise ValueError("Provide either --init_image or --init_video for I2V conditioning")

    # Validate V-JEPA parameters for guidance and rejection sampling
    if args.sampling_method in ['guidance', 'rejection']:
        if args.vjepa_context_frames >= 16:
            raise ValueError(f"vjepa_context_frames ({args.vjepa_context_frames}) must be less than 16 (torch V-JEPA frames_per_clip)")
        if 16 > args.num_frames:
            raise ValueError(f"torch V-JEPA requires at least 16 frames, but num_frames is {args.num_frames}")
        if args.slice_stride <= 0:
            raise ValueError(f"slice_stride ({args.slice_stride}) must be positive")

    # Prepare prompts and negative prompt for non-batch mode
    if not args.batch_json:
        if args.prompt is not None:
            prompts = [args.prompt]
            # Define a negative prompt suitable for CogVideoX
            if "CogVideoX" in args.model_id:
                negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"
            elif "Cosmos" in args.model_id:
                negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
        else:
            if not args.prompt_file:
                raise ValueError("Provide either --prompt or --prompt_file")
            prompts, negative_prompt = get_prompts(args.prompt_file, args)

    # Generate simple experiment name
    experiment_name = get_simple_experiment_name(args)
    
    # Non-batch: chunk the prompts for distributed processing
    if not args.batch_json:
        chunked_prompts = chunk_prompts(prompts, args.num_gpus, args.gpu_idx)

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
    if not args.batch_json:
        print(f"Prompts assigned to this GPU: {len(chunked_prompts)}")
    else:
        print(f"Running in batch_json mode. Sharding entries by index modulo {args.num_gpus} (this worker idx={args.gpu_idx}).")
    
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
    
    print(f"{'='*60}\n")

    # Initialize pipeline once per process
    pipe = init_pipeline(args)
    
    # Initialize V-JEPA models for rejection sampling if enabled
    vjepa_models = init_vjepa_models(args)

    if not args.batch_json:
        # Single or multi-prompt mode using provided init image/video
        init_frame = load_first_frame(args.init_image, args.init_video)
        generate_videos(pipe, args, init_frame, chunked_prompts, negative_prompt, experiment_name, fps=args.fps, vjepa_models=vjepa_models)
    else:
        # Batch JSON mode: load tasks, shard by index, and process this shard
        with open(args.batch_json, 'r') as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError("--batch_json must contain a JSON array of entries")

        # Determine dataset mode and base directory
        dataset_mode = args.dataset_mode if args.dataset_mode != 'auto' else detect_dataset_mode(args.batch_json)
        base_dir = args.base_dir or ''
        
        print(f"Dataset mode: {dataset_mode}")
        if dataset_mode == 'physics_iq':
            print("Using Physics-IQ mode: absolute input paths, relative output paths")
        else:
            print("Using DreamBench mode: relative paths with base_dir")
        
        # Use same chunking mechanism as vanilla/guidance methods for consistent ordering
        chunked_entries = chunk_prompts(entries, args.num_gpus, args.gpu_idx)
        
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
            input_video_abs, input_image_abs, output_video_abs = resolve_paths(
                input_video, input_image, output_video, base_dir, dataset_mode
            )

            # Prepare per-item init frame
            if "5frame" not in args.batch_json:
                init_frame = load_first_frame(input_image_abs, input_video_abs)
            else:
                init_frame = load_video(input_video_abs)

            # Prepare prompts list and negative prompt
            per_item_prompts = [prompt]
            negative_prompt = (
                "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"
            )

            # Save under the configured output folder and experiment name.
            # Use the JSON-provided output filename, but place it inside
            # <output_folder>/<experiment>/ to match project structure.
            args.output_path = None
            args.output_filename = os.path.basename(output_video_abs)
            
            args.init_image = input_image_abs
            args.init_video = input_video_abs

            print(f"[GPU {args.gpu_idx}] {os.path.basename(output_video_abs)}")
            generate_videos(pipe, args, init_frame, per_item_prompts, negative_prompt, experiment_name, fps=args.fps, vjepa_models=vjepa_models)
            processed_count += 1

        print(f"Processed {processed_count} entries on GPU index {args.gpu_idx}.")

if __name__ == "__main__":
    main()
