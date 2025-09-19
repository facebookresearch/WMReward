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
    
    # Add SMC parameters
    if args.sampling_method == 'smc':
        metadata["parameters"].update({
            "smc_num_particles": getattr(args, 'smc_num_particles', 16),
            "smc_beta_const": getattr(args, 'smc_beta_const', 24.0),
            "smc_ess_threshold": getattr(args, 'smc_ess_threshold', 0.97),
            "smc_early_frac": getattr(args, 'smc_early_frac', 0.10),
            "smc_late_frac": getattr(args, 'smc_late_frac', 0.70),
            "smc_step_stride": getattr(args, 'smc_step_stride', 5),
            "smc_potential_mode": getattr(args, 'smc_potential_mode', 'max'),
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
            f"_cfg{args.cfg_scale}_gsp{getattr(args, 'guidance_step_pattern', '')}_glp{getattr(args, 'guidance_lr_pattern', '')}_gf{getattr(args, 'guidance_frequency', '')}_{loss_mode}"
        )
        
        # Add vjepa variant if not default
        if vjepa_short:
            name += f"_{vjepa_short}"
    elif args.sampling_method == 'rejection':
        name = f"{args.sampling_method}_{version}_f{args.num_frames}_s{args.num_inference_steps}_cfg{args.cfg_scale}"
    elif args.sampling_method == 'smc':
        name = (
            f"smc_{version}_f{args.num_frames}_s{args.num_inference_steps}"
            f"_cfg{args.cfg_scale}_N{getattr(args, 'smc_num_particles', 16)}"
        )

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
        from pipelines.pipeline_cosmos_image2video_savemem import Cosmos2VideoToWorldPipeline
        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to("cuda")
        pipe = pipe.to("cuda")
        pipe.transformer.enable_gradient_checkpointing()

        # pipe.enable_sequential_cpu_offload() # it needs to be enabled for the model to run on A6000
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe

def init_vjepa_models(args):
    """Initialize V-JEPA models for rejection sampling evaluation."""
    if args.sampling_method not in ['rejection', 'smc']:
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
            additional_inputs={
                "vjepa_variant": getattr(args, 'vjepa_variant', "vit_giant"),
                "vjepa_img_size": int(getattr(args, 'vjepa_img_size', 256)),
                "vjepa_masking_mode": str(getattr(args, 'vjepa_masking_mode', 'causal')),
                "vjepa_context_frames": context_frames,
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
def _decode_full(pipe, latents):
    latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    frames = pipe.decode_latents(latents)
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
        raise RuntimeError(f"Unexpected decoded shape/type: {type(frames)} {getattr(frames,'shape',None)}")
    frames = _ensure_btchw(frames)
    frames = _to_minus1_1(frames)
    return frames

def _ess_normed(weights: torch.Tensor) -> float:
    return float(1.0 / (weights.pow(2).sum() + 1e-8))

def _weight_entropy_bits(weights: torch.Tensor) -> float:
    w = weights.clamp_min(1e-12)
    H = -torch.sum(w * torch.log2(w))
    return float(H)

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
            img_size=getattr(args, 'vjepa_img_size', 256),
            window_size=int(getattr(args, 'slice_window_size', 16)),
            loss_exp=2,
            masking_mode=str(getattr(args, 'vjepa_masking_mode', 'causal')),
            context_frames=int(getattr(args, 'vjepa_context_frames', 8)),
            mask_ratio=float(getattr(args, 'vjepa_mask_ratio', 0.75)),
            spatial_pred_mask_scale=None,
            temporal_pred_mask_scale=None,
            aspect_ratio=None,
            npred=None,
            max_context_frames_ratio=None,
            is_vae_output=True,
            seed=int(getattr(args, 'seed', 42)),
            stride=int(getattr(args, 'slice_stride', 8)),
            mode=str(getattr(args, 'loss_mode', 'max')),
        )
        out[i] = float(loss)
    return out

def smc_sample(pipe, args, init_frame, prompt, negative_prompt, vjepa_models, generator=None):
    if "CogVideoX" not in args.model_id:
        raise NotImplementedError("SMC sampling is currently implemented for CogVideoX only.")

    encoder, target_encoder, predictor = vjepa_models if vjepa_models else (None, None, None)
    if encoder is None or target_encoder is None or predictor is None:
        raise RuntimeError("V-JEPA models must be loaded for SMC; set sampling_method=smc to trigger loading.")

    steps = int(args.num_inference_steps)
    num_particles = int(getattr(args, 'smc_num_particles', 16))
    beta_const = float(getattr(args, 'smc_beta_const', 24.0))
    ess_threshold = float(getattr(args, 'smc_ess_threshold', 0.97))
    early_frac = float(getattr(args, 'smc_early_frac', 0.20))
    late_frac = float(getattr(args, 'smc_late_frac', 0.90))
    step_stride = int(getattr(args, 'smc_step_stride', 5))

    start = int(round(steps * early_frac))
    end = int(round(steps * late_frac))
    check_steps = list(range(start, end, max(1, step_stride)))
    freeze_after_step = end

    generators = [torch.Generator(device="cuda").manual_seed(int(args.seed) + i) for i in range(num_particles)]
    weights = torch.full((num_particles,), 1.0 / num_particles, device="cuda", dtype=torch.float32)
    running_max = torch.full((num_particles,), -torch.inf, device="cuda", dtype=torch.float32)
    lineage = torch.arange(num_particles, device="cuda")
    product_of_potentials = torch.ones(num_particles, device="cuda", dtype=torch.float32)
    population_rs = torch.zeros(num_particles, device="cuda", dtype=torch.float32)
    frozen_idx = None

    @torch.inference_mode()
    def fk_callback(pipe_obj, step: int, timestep: int, callback_kwargs: dict, **_):
        nonlocal running_max, frozen_idx, weights, product_of_potentials, population_rs, lineage

        latents = callback_kwargs.get("latents", None)
        if latents is None:
            raise RuntimeError("Pipeline must expose 'latents' via callback_on_step_end_tensor_inputs=['latents'].")

        if step >= freeze_after_step:
            if frozen_idx is None:
                vids_full = _decode_full(pipe_obj, latents)
                surprise_t = _vjepa_surprise_batch(vids_full, encoder, target_encoder, predictor, args)
                frozen_idx = int(torch.argmin(surprise_t).item())
                print(f"[FK] FREEZE at step {step}: locking particle {frozen_idx}")

            idx = torch.full((latents.shape[0],), frozen_idx, device=latents.device, dtype=torch.long)
            latents = latents.index_select(0, idx)
            weights.fill_(1.0 / weights.numel())
            return {"latents": latents}

        if step in check_steps:
            vids_full = _decode_full(pipe_obj, latents)  # (B,3,T,H,W) in [-1,1]
            surprise_t = _vjepa_surprise_batch(vids_full, encoder, target_encoder, predictor, args)
            phi_t = 1.0 - surprise_t

            running_max = torch.maximum(running_max, phi_t)
            phi = running_max

            pot_term = torch.exp((beta_const * phi).clamp(min=-60.0, max=60.0))
            product_of_potentials = product_of_potentials * pot_term
            population_rs = phi

            last_check_step = check_steps[-1] if len(check_steps) > 0 else -1
            is_final_point = (step == last_check_step) or (step == (steps - 1))
            if is_final_point:
                w = torch.exp((beta_const * population_rs).clamp(min=-60.0, max=60.0)) / (product_of_potentials + 1e-8)
                w = torch.clamp(w, 0.0, 1e10)
                w[w.isnan()] = 0.0

                normalized_w = w / (w.sum() + 1e-8)
                ess = 1.0 / (normalized_w.pow(2).sum() + 1e-8)
                print(f"      [final-correction] ESS={float(ess):.3f}")

                if ess < 0.5 * num_particles:
                    print(f"      [final-correction] RESAMPLE at step {step} with ESS={float(ess):.3f}")
                    idx = torch.multinomial(w, num_samples=w.numel(), replacement=True)
                    latents = latents.index_select(0, idx)
                    weights.fill_(1.0 / w.numel())
                    running_max = running_max.index_select(0, idx)
                    product_of_potentials = product_of_potentials.index_select(0, idx)
                    population_rs = population_rs.index_select(0, idx)
                    lineage.copy_(lineage.index_select(0, idx))
                else:
                    weights.copy_(normalized_w)

                return {"latents": latents}

            z = beta_const * phi
            z = z - z.max()
            incr = torch.exp(z.clamp(min=-60.0))
            new_w = weights * incr
            new_w = new_w / (new_w.sum() + 1e-8)

            ess_over_n = _ess_normed(new_w) / new_w.numel()
            ent_bits = _weight_entropy_bits(new_w)
            best_loss = float(surprise_t.min())
            best_idx = int(surprise_t.argmin())
            worst_loss = float(surprise_t.max())
            mean_loss = float(surprise_t.mean())
            std_loss = float(surprise_t.std(unbiased=False))
            do_resample = (ess_over_n < ess_threshold)

            print(
                f"[FK] step={step:02d} (t={int(timestep)}), "
                f"loss min/mean±std/max= {best_loss:.4f} / {mean_loss:.4f}±{std_loss:.4f} / {worst_loss:.4f}, "
                f"beta={beta_const:.1f}, ESS/N={ess_over_n:.3f}, H(bits)={ent_bits:.2f}, resample={do_resample}"
            )

            if do_resample:
                idx = torch.multinomial(new_w, num_samples=new_w.numel(), replacement=True)
                latents = latents.index_select(0, idx)
                weights.fill_(1.0 / new_w.numel())
                running_max = running_max.index_select(0, idx)
                product_of_potentials = product_of_potentials.index_select(0, idx)
                population_rs = population_rs.index_select(0, idx)
                lineage.copy_(lineage.index_select(0, idx))
                print(f"      RESAMPLE (multinomial)! -> weights reset")
            else:
                weights.copy_(new_w)
                print(f"      no resample (ESS/N={ess_over_n:.3f} ≥ {ess_threshold:.3f})")

            return {"latents": latents}

        return {}

    zero_steps = [0] * steps
    zero_lrs = [0.0] * steps
    out = pipe(
        video=[init_frame] * num_particles,
        prompt=[prompt] * num_particles,
        negative_prompt=None,
        num_frames=args.num_frames,
        num_inference_steps=steps,
        guidance_scale=args.cfg_scale,
        use_dynamic_cfg=True,
        num_videos_per_prompt=1,
        eta=0.01,
        generator=generators,
        callback_on_step_end=fk_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        guidance_step=zero_steps,
        guidance_lr=zero_lrs,
        guidance_frequency=1,
        additional_inputs=None,
        travel_time=(0, 0),
    )

    if not out.frames:
        raise RuntimeError("No frames returned from SMC inference!")

    vids_btchw_list = []
    for particle_frames in out.frames:
        frame_tensors = []
        for pil_frame in particle_frames:
            frame_np = np.array(pil_frame)
            frame_tensor = torch.from_numpy(frame_np).float().permute(2, 0, 1)
            frame_tensor = (frame_tensor / 255.0) * 2.0 - 1.0
            frame_tensors.append(frame_tensor)
        particle_video = torch.stack(frame_tensors, dim=0).permute(1, 0, 2, 3)
        vids_btchw_list.append(particle_video)

    vids_btchw = torch.stack(vids_btchw_list, dim=0).cuda()
    final_surprise = _vjepa_surprise_batch(vids_btchw, encoder, target_encoder, predictor, args)
    best_idx = int(final_surprise.argmin().item())
    print(f"[FK] FINAL best particle: {best_idx}  V-JEPA loss: {float(final_surprise[best_idx]):.6f}")
    return out.frames[best_idx]

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
            
            # Generate multiple samples and select best based on V-JEPA loss
            candidate_frames = []
            candidate_losses = []
            
            for sample_idx in range(args.rejection_samples):
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
                            seed=args.seed,
                            stride=stride,
                            mode=getattr(args, 'loss_mode', 'max')
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

        elif args.sampling_method == 'smc':
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            frames = smc_sample(
                pipe=pipe,
                args=args,
                init_frame=init_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                vjepa_models=vjepa_models,
                generator=generator
            )

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
                       choices=['vanilla', 'guidance', 'rejection', 'smc'],
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
    parser.add_argument('--vjepa_mask_ratio', type=float, default=0.5)
    parser.add_argument('--slice_window_size', type=int, default=16)
    parser.add_argument('--slice_stride', type=int, default=4)
    parser.add_argument('--vae_decode_scale', type=float, default=0.7, help='VAE decode scale factor.')
    parser.add_argument('--loss_mode', type=str, default='max', choices=['mean', 'max'], help='V-JEPA loss aggregation mode.')
    
    # Rejection sampling parameters (only used when sampling_method='rejection')
    parser.add_argument('--rejection_samples', type=int, default=3, 
                       help='Number of samples to generate for rejection sampling.')

    # SMC parameters (only used when sampling_method='smc')
    parser.add_argument('--smc_num_particles', type=int, default=16, help='Number of SMC particles.')
    parser.add_argument('--smc_beta_const', type=float, default=24.0, help='FK potential temperature beta.')
    parser.add_argument('--smc_ess_threshold', type=float, default=0.97, help='Resample when ESS/N < threshold.')
    parser.add_argument('--smc_early_frac', type=float, default=0.10, help='Start of mid-window as frac of steps.')
    parser.add_argument('--smc_late_frac', type=float, default=0.70, help='End of mid-window as frac of steps.')
    parser.add_argument('--smc_step_stride', type=int, default=5, help='Check every k steps in mid-window.')

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
    
    if args.sampling_method == 'smc':
        print(f"SMC steering enabled:")
        print(f"  - Particles: {args.smc_num_particles}")
        print(f"  - Beta: {args.smc_beta_const}")
        print(f"  - ESS threshold: {args.smc_ess_threshold}")
        print(f"  - Mid-window: [{args.smc_early_frac}, {args.smc_late_frac}), stride {args.smc_step_stride}")
    
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
