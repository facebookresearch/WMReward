import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dataclasses import dataclass
from pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
# from pipeline.diffusers_pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video, load_image
import os

from utils import compute_vjepa_loss_sliding_window, load_vjepa_models_torchhub

# ===================== minimal defaults ======================
MODEL_ID = "THUDM/CogVideoX-5b-I2V"
PROMPT = ("Use the left hand to pick up dark green cucumber from on circular gray mat "
          "to above beige bowl. Ultrarealistic, cinematic.")
INIT_IMAGE = "/home/yjianhao/project/frame-guidance/dream_gen_benchmark/gr1_object/0_Use the left hand to pick up dark green cucumber from on circular gray mat to above beige bowl..png"

NUM_FRAMES     = 49
NUM_STEPS      = 50
GUIDANCE_SCALE = 6.0
FPS            = 8

# ===================== FK/SMC (mid-window; freeze after late_frac) ======================
N_PARTICLES    = 4
BETA_CONST     = 24.0       # temperature β (increase for more aggressive pruning)
EARLY_FRAC     = 0.30       # start of mid-window (fraction of steps)
LATE_FRAC      = 0.90       # freeze from this fraction of steps (end of SMC, start single particle)  
STEP_STRIDE    = 5          # check and resample every k steps within [EARLY_FRAC, LATE_FRAC) (deterministic)

POTENTIAL_MODE = "current"  # Use current step reward (no running max)
SEED0          = 42

# =============== DEBUG CONFIG =================
DEBUG_TOPK       = 4
PRINT_WEIGHTS    = False
PRINT_IDX_HEAD   = 10
LINEAGE_HIST_MAX = 10

# V-JEPA call defaults
VJEPA_CFG = dict(
    vjepa_variant="vit_giant",
    vjepa_img_size=256,
    vjepa_masking_mode="causal",
    vjepa_context_frames=8,
    vjepa_mask_ratio=0.75,
    slice_window_size=16,
    slice_stride=8,
    loss_mode="max",
)

# ===================== helpers ======================
def build_seq(pattern: str, steps: int, is_float: bool):
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

def ensure_bcthw(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 5, f"expected 5D video, got {x.shape}"
    if x.shape[1] == 3:          # (B,C,T,H,W)
        return x
    if x.shape[2] == 3:          # (B,T,C,H,W)
        return x.permute(0, 2, 1, 3, 4).contiguous()
    if x.shape[-1] == 3:         # (B,T,H,W,C)
        return x.permute(0, 4, 1, 2, 3).contiguous()
    raise RuntimeError(f"Cannot infer channel dim in {x.shape}; expected channel==3 at dim 1/2/-1.")

def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    xmin, xmax = float(x.min()), float(x.max())
    if -0.05 <= xmin and xmax <= 1.05:          # [0,1] → [-1,1]
        return x * 2.0 - 1.0
    if 0.0 <= xmin and xmax <= 255.0:           # [0,255] → [-1,1]
        return (x / 127.5) - 1.0
    return x

@torch.inference_mode()
def decode_full(pipe, latents):
    """
    CogVideoX decode to (B,3,T,H,W) in [-1,1] with chunked processing.
    """
    latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    B = latents.shape[0]
    frames = pipe.decode_latents(latents)
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
        raise RuntimeError(f"Unexpected decoded shape/type: {type(frames)} {getattr(frames,'shape',None)}")
    frames = ensure_bcthw(frames)
    frames = to_minus1_1(frames)
    return frames

def ess_normed(w: torch.Tensor) -> float:
    return float(1.0 / (w.pow(2).sum() + 1e-8))

def weight_entropy_bits(w: torch.Tensor) -> float:
    w_ = w.clamp_min(1e-12)
    H = -torch.sum(w_ * torch.log2(w_))
    return float(H)

@torch.inference_mode()
def vjepa_surprise_batch(vids_btchw: torch.Tensor,
                         encoder, target_encoder, predictor) -> torch.Tensor:
    vids_btchw = ensure_bcthw(vids_btchw).to(dtype=torch.float32)
    B = vids_btchw.shape[0]
    out = torch.empty(B, device=vids_btchw.device, dtype=torch.float32)
    
    for i in range(0, B):
        loss = compute_vjepa_loss_sliding_window(
            video_tensor=vids_btchw[i:i+1],
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            img_size=VJEPA_CFG["vjepa_img_size"],
            window_size=VJEPA_CFG["slice_window_size"],
            loss_exp=2,
            masking_mode=VJEPA_CFG["vjepa_masking_mode"],
            context_frames=VJEPA_CFG["vjepa_context_frames"],
            mask_ratio=VJEPA_CFG["vjepa_mask_ratio"],
            spatial_pred_mask_scale=None,
            temporal_pred_mask_scale=None,
            aspect_ratio=None,
            npred=None,
            max_context_frames_ratio=None,
            is_vae_output=True,
            seed=42,
            stride=VJEPA_CFG["slice_stride"],
            mode=VJEPA_CFG["loss_mode"],
        )
        out[i] = float(loss)
    return out  # (B,)

# ===================== main ======================
def main():
    # --- 1) Pipeline
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    pipe.scheduler = CogVideoXDDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe.to("cuda")
    if hasattr(pipe, "vae"):
        try:
            pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
        except Exception:
            pass

    init_image = load_image(INIT_IMAGE).convert("RGB")

    # --- 2) Load V-JEPA models
    encoder, target_encoder, predictor, _ = load_vjepa_models_torchhub(
        VJEPA_CFG["vjepa_variant"]
    )
    encoder.eval().cuda(); target_encoder.eval().cuda(); predictor.eval().cuda()

    # --- 3) Particles, weights, current rewards, lineage
    gens = [torch.Generator(device="cuda").manual_seed(SEED0 + i) for i in range(N_PARTICLES)]
    weights     = torch.full((N_PARTICLES,), 1.0 / N_PARTICLES, device="cuda", dtype=torch.float32)
    lineage     = torch.arange(N_PARTICLES, device="cuda")
    # Track current rewards only (no accumulation across steps)
    population_rs = torch.zeros(N_PARTICLES, device="cuda", dtype=torch.float32)

    # --- 3b) Mid-window checks + freeze boundary
    start = int(round(NUM_STEPS * EARLY_FRAC))
    end   = int(round(NUM_STEPS * LATE_FRAC))
    CHECK_STEPS = list(range(start, end, max(1, STEP_STRIDE)))
    FREEZE_AFTER_STEP = end
    frozen_idx = None

    print(f"[FK] checkpoints (0-based): {CHECK_STEPS}")
    print(f"[FK] freeze from step >= {FREEZE_AFTER_STEP}")
    print(f"[FK] N={N_PARTICLES}  beta={BETA_CONST}  mode={POTENTIAL_MODE}  (deterministic resample every {STEP_STRIDE} steps)")
    print(f"[FK] Single-GPU FK/SMC steering with {N_PARTICLES} particles in one batch")

    # --- 4) FK/SMC callback
    @torch.inference_mode()
    def fk_callback(pipe_obj, step: int, timestep: int, callback_kwargs: dict, **_):
        nonlocal frozen_idx, weights, population_rs

        latents = callback_kwargs.get("latents", None)  # (B, F, C, H, W)
        if latents is None:
            raise RuntimeError("Pipeline must expose 'latents' via callback_on_step_end_tensor_inputs=['latents'].")

        # ---- Freeze region: lock to the best particle; no more scoring/resampling ----
        if step >= FREEZE_AFTER_STEP:
            if frozen_idx is None:
                # Score once at the boundary for a precise pick
                vids_full = decode_full(pipe_obj, latents)
                surprise_t = vjepa_surprise_batch(vids_full, encoder, target_encoder, predictor)  # lower is better
                frozen_idx = int(torch.argmin(surprise_t).item())
                print(f"[FK] FREEZE at step {step}: locking to particle {frozen_idx} only")

                # Keep only the best particle (reduce batch size to 1 for memory efficiency during guidance)
                latents = latents[frozen_idx:frozen_idx+1]
                
                # Update weights to single particle
                weights = torch.ones(1, device="cuda", dtype=torch.float32)
                return {"latents": latents}
            
            # After the first freeze step, don't modify latents anymore - let pipeline handle batch size
            return {}
        # -------------------------------------------------------------------------------------------

        if step in CHECK_STEPS:
            # 1) Decode and score
            vids_full = decode_full(pipe_obj, latents)  # (B,3,T,H,W) in [-1,1]
            surprise_t = vjepa_surprise_batch(vids_full, encoder, target_encoder, predictor)  # (B,)
            phi_t = 1.0 - surprise_t     # higher is better

            # 2) Potential: use current step reward directly
            phi = phi_t  # Use current step reward, not running max

            # Use current reward only (no accumulation)
            population_rs = phi

            # If this is the final point, use current reward for resampling
            last_check_step = CHECK_STEPS[-1] if len(CHECK_STEPS) > 0 else -1
            is_final_point = (step == last_check_step) or (step == (NUM_STEPS - 1))
            if is_final_point:
                # Use current reward directly for final resampling
                w = torch.exp((BETA_CONST * population_rs).clamp(min=-60.0, max=60.0))
                w = torch.clamp(w, 0.0, 1e10)
                w[w.isnan()] = 0.0

                normalized_w = w / (w.sum() + 1e-8)
                print(f"      [final-correction] Final step: always resample regardless of stride")
                
                # Always resample at final step (deterministic)
                idx = torch.multinomial(w, num_samples=w.numel(), replacement=True)
                latents = latents.index_select(0, idx)
                weights.fill_(1.0 / w.numel())
                population_rs = population_rs.index_select(0, idx)
                lineage.copy_(lineage.index_select(0, idx))

                return {"latents": latents}

            # 3) Weights update (non-final steps)
            z = BETA_CONST * phi
            z = z - z.max()
            incr = torch.exp(z.clamp(min=-60.0))
            new_w = weights * incr
            new_w = new_w / (new_w.sum() + 1e-8)

            # 4) Deterministic resampling every STEP_STRIDE steps
            best_loss, best_idx = float(surprise_t.min()), int(surprise_t.argmin())
            worst_loss          = float(surprise_t.max())
            mean_loss           = float(surprise_t.mean())
            std_loss            = float(surprise_t.std(unbiased=False))

            print(
                f"[FK] step={step:02d} (t={int(timestep)}), "
                f"loss min/mean±std/max= {best_loss:.4f} / {mean_loss:.4f}±{std_loss:.4f} / {worst_loss:.4f}, "
                f"beta={BETA_CONST:.1f}, stride={STEP_STRIDE} (deterministic resample)"
            )
            if DEBUG_TOPK > 0:
                topk_phi = torch.topk(phi, k=min(DEBUG_TOPK, phi.numel()))
                print(f"      top{topk_phi.values.numel()} φ (current) idx={topk_phi.indices.tolist()} "
                      f"vals={[f'{v:.4f}' for v in topk_phi.values.tolist()]} (best_loss idx={best_idx})")
                topk_w = torch.topk(new_w, k=min(DEBUG_TOPK, new_w.numel()))
                print(f"      top{topk_w.values.numel()} w   idx={topk_w.indices.tolist()} "
                      f"vals={[f'{v:.3f}' for v in topk_w.values.tolist()]}")

            if PRINT_WEIGHTS:
                print(f"      w = {[float(x) for x in new_w.tolist()]}")

            # 5) Deterministic resampling (always resample at checkpoint steps)
            # --- multinomial ---
            idx = torch.multinomial(new_w, num_samples=new_w.numel(), replacement=True)
            latents = latents.index_select(0, idx)

            # reset weights; propagate lineage
            weights.fill_(1.0 / new_w.numel())
            population_rs = population_rs.index_select(0, idx)
            lineage.copy_(lineage.index_select(0, idx))

            # lineage histogram (debug)
            print(f"      RESAMPLE (multinomial, every {STEP_STRIDE} steps)! idx[:{PRINT_IDX_HEAD}]={idx[:PRINT_IDX_HEAD].tolist()} -> weights reset")
            counts = torch.bincount(lineage, minlength=N_PARTICLES).cpu().numpy()
            head_bins = min(LINEAGE_HIST_MAX, len(counts))
            hist_str = ", ".join([f"{i}:{int(c)}" for i, c in enumerate(counts[:head_bins]) if c > 0])
            tail_nonzero = [(i, int(c)) for i, c in enumerate(counts[head_bins:]) if c > 0]
            if hist_str or tail_nonzero:
                print(f"      lineage copies (orig_seed_id:count) head -> {hist_str}" +
                      (f" ... +{len(tail_nonzero)} more" if tail_nonzero else ""))

            return {"latents": latents}

        return {}  # outside mid-window: no-op

    # --- 5) Run steered sampling (single GPU, batched particles)
    out = pipe(
        video=[init_image] * N_PARTICLES,
        prompt=[PROMPT] * N_PARTICLES,
        # negative_prompt=[NEG_PROMPT] * N_PARTICLES,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        use_dynamic_cfg=True,
        num_videos_per_prompt=1,
        eta=1.0,
        generator=gens,
        callback_on_step_end=fk_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        guidance_step=build_seq("0x50", NUM_STEPS, is_float=False),
        guidance_lr=build_seq("0x50", NUM_STEPS, is_float=True),
        guidance_frequency=1,
        additional_inputs=None,
    )
    # out = pipe(
    #     video=[init_image] * N_PARTICLES,
    #     prompt=[PROMPT] * N_PARTICLES,
    #     # negative_prompt=[NEG_PROMPT] * N_PARTICLES,
    #     num_frames=NUM_FRAMES,
    #     num_inference_steps=NUM_STEPS,
    #     guidance_scale=GUIDANCE_SCALE,
    #     use_dynamic_cfg=True,
    #     num_videos_per_prompt=1,
    #     eta=1.0,
    #     generator=gens,
    #     callback_on_step_end=fk_callback,
    #     callback_on_step_end_tensor_inputs=["latents"],
    #     guidance_step=build_seq("0x25,1x25", NUM_STEPS, is_float=False),
    #     guidance_lr=build_seq("0x25,0.003x25", NUM_STEPS, is_float=True),
    #     guidance_frequency=3,
    #     additional_inputs=None,
    # )

    # --- 6) Final pick via V-JEPA (FAIL HARD for debugging)
    if not out.frames:
        raise RuntimeError("No frames returned from distributed inference!")
    
    vids_btchw_list = []
    for i, particle_frames in enumerate(out.frames):  # each particle's frames (list of PIL images)
        if particle_frames is None:
            raise RuntimeError(f"Particle {i} returned None frames!")
        
        frame_tensors = []
        for j, pil_frame in enumerate(particle_frames):
            if pil_frame is None:
                raise RuntimeError(f"Particle {i}, frame {j} is None!")
            frame_np = np.array(pil_frame)
            frame_tensor = torch.from_numpy(frame_np).float().permute(2, 0, 1)  # (C,H,W)
            frame_tensor = (frame_tensor / 255.0) * 2.0 - 1.0
            frame_tensors.append(frame_tensor)
        particle_video = torch.stack(frame_tensors, dim=0).permute(1, 0, 2, 3)  # (C,T,H,W)
        vids_btchw_list.append(particle_video)

    vids_btchw = torch.stack(vids_btchw_list, dim=0).cuda()  # (B,C,T,H,W) in [-1,1]
    final_surprise = vjepa_surprise_batch(vids_btchw, encoder, target_encoder, predictor)
    best_idx = int(final_surprise.argmin().item())
    print(f"[FK] FINAL best particle: {best_idx}  V-JEPA loss: {float(final_surprise[best_idx]):.6f}")

    export_to_video(out.frames[best_idx], "output_fk_vjepa.mp4", fps=FPS)
    print("Saved: output_fk_vjepa.mp4")

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"[FK] Single-GPU FK/SMC steering")
    main()
