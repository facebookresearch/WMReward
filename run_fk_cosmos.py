# run_fk_steering_cosmos_fk.py
# Feynman–Kac steering for Cosmos
# - Potential: diff / sum / max (you set POTENTIAL_MODE below)
# - Resampling: ESS-only (no forced first resample), mid-window checks
# - Freeze after late_frac: keep one latent for the rest of the trajectory

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dataclasses import dataclass

from pipelines.pipeline_cosmos_image2video_savemem import Cosmos2VideoToWorldPipeline
from schedulers.flow_discrete_euler import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from utils import compute_vjepa_loss_sliding_window, load_vjepa_models_torchhub

# ===================== minimal defaults ======================
MODEL_ID    = "nvidia/Cosmos-Predict2-2B-Video2World"
PROMPT      = "Use the right hand to pick up orange carrot from center of table to lower white shelf."
NEG_PROMPT  = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"
INIT_IMAGE  = "/home/yjianhao/project/frame-guidance/dream_gen_benchmark/gr1_object/4_Use the right hand to pick up orange carrot from center of table to lower white shelf..png"

HEIGHT, WIDTH = 704, 1280
NUM_FRAMES    = 93
NUM_STEPS     = 35
GUIDANCE_SCALE = 7.0
FPS = 16

# ===================== FK/SMC ======================
N_PARTICLES    = 4
BETA_CONST     = 24.0     # temperature β
ESS_THRESHOLD  = 0.99     # resample when ESS/N < this

# Mid-window checks
EARLY_FRAC     = 0.1      # start of check window (fraction of steps)
LATE_FRAC      = 0.7      # end   of check window (fraction of steps) -> we freeze from here
STEP_STRIDE    = 5       # check every k steps within [EARLY_FRAC, LATE_FRAC)

# Potential mode: "diff", "sum", or "max"
POTENTIAL_MODE = "max"

SEED0          = 42

# =============== DEBUG CONFIG =================
DEBUG_TOPK       = 4          # show top-k particles by φ_t and by weight
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
def _save_intermediate_video(decoded_video_bcthw: torch.Tensor, path: str, fps: int = 8) -> None:
    x = decoded_video_bcthw.detach().cpu()
    x = ((x.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    B, C, T, H, W = x.shape
    frames = []
    for tt in range(T):
        frame = (x[0, :, tt] * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
        frames.append(Image.fromarray(frame))
    export_to_video(frames, path, fps=fps)
    del x, frames
    torch.cuda.empty_cache()

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
    Cosmos/WAN VAE decodes 5D latents: expects (B, C, T, H, W).
    Callback supplies (B, F, C, H, W) -> (B, C, F, H, W), re-scale by VAE stats, decode.
    Returns (B, 3, T, H_out, W_out) in [-1, 1].
    """
    z = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)

    # Re-scale to WAN VAE domain
    sigma_data = float(getattr(getattr(pipe, "scheduler", object()), "config", object()).__dict__.get("sigma_data", 1.0))
    z_dim = int(getattr(pipe.vae.config, "z_dim", z.shape[1]))
    lm = getattr(pipe.vae.config, "latents_mean", [0.0] * z_dim)
    ls = getattr(pipe.vae.config, "latents_std",  [1.0] * z_dim)
    latents_mean = torch.tensor(lm, device=z.device, dtype=z.dtype).view(1, z_dim, 1, 1, 1)
    latents_std  = torch.tensor(ls, device=z.device, dtype=z.dtype).view(1, z_dim, 1, 1, 1)

    z = z * latents_std / sigma_data + latents_mean

    # WAN decode expects 5D (B,C,T,H,W)
    frames = pipe.vae.decode(z.to(pipe.vae.dtype), return_dict=False)[0]

    # (Optional) write one sample video for debugging
    _save_intermediate_video(frames, f"./temp/fk_steering_cosmos.mp4", fps=FPS)

    frames = ensure_bcthw(frames)
    frames = to_minus1_1(frames)
    return frames

def ess_normed(w: torch.Tensor) -> float:
    return float(1.0 / (w.pow(2).sum() + 1e-8))

def weight_entropy_bits(w: torch.Tensor) -> float:
    w_ = w.clamp_min(1e-12)
    H = -torch.sum(w_ * torch.log2(w_))
    return float(H)

def stratified_indices(w_norm: torch.Tensor):
    B = w_norm.numel()
    u0 = torch.rand((), device=w_norm.device) / B
    cdf = torch.cumsum(w_norm, dim=0)
    pos = u0 + torch.arange(B, device=w_norm.device) / B
    return torch.searchsorted(cdf, pos.clamp(max=1 - 1e-8))

@torch.inference_mode()
def vjepa_surprise_batch(vids_btchw: torch.Tensor, encoder, target_encoder, predictor) -> torch.Tensor:
    vids_btchw = ensure_bcthw(vids_btchw).to(dtype=torch.float32)
    B = vids_btchw.shape[0]
    out = torch.empty(B, device=vids_btchw.device, dtype=torch.float32)
    for i in range(B):
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
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe.scheduler.config.stochastic_sampling = True
    pipe.to("cuda")

    pipe.transformer.enable_gradient_checkpointing()
    pipe.vae.enable_tiling(); pipe.vae.enable_slicing()

    init_image = load_image(INIT_IMAGE).convert("RGB")

    # --- 2) Load V-JEPA models
    encoder, target_encoder, predictor, _ = load_vjepa_models_torchhub(VJEPA_CFG["vjepa_variant"])
    encoder.eval().cuda(); target_encoder.eval().cuda(); predictor.eval().cuda()

    # --- 3) Particles, weights, potentials, lineage
    gens = [torch.Generator(device="cuda").manual_seed(SEED0 + i) for i in range(N_PARTICLES)]
    weights  = torch.full((N_PARTICLES,), 1.0 / N_PARTICLES, device="cuda", dtype=torch.float32)
    last_phi = torch.zeros(N_PARTICLES, device="cuda", dtype=torch.float32)  # for "diff"

    # for "sum" and "max" potentials
    running_sum = torch.zeros_like(last_phi)
    running_max = torch.full_like(last_phi, -torch.inf)

    lineage  = torch.arange(N_PARTICLES, device="cuda")
    
    # Track product of potentials for final-step correction
    product_of_potentials = torch.ones(N_PARTICLES, device="cuda", dtype=torch.float32)
    population_rs = torch.zeros(N_PARTICLES, device="cuda", dtype=torch.float32)

    # --- 3b) Mid-window checkpoints + freeze boundary
    start = int(round(NUM_STEPS * EARLY_FRAC))   # e.g., 0.20 * T
    end   = int(round(NUM_STEPS * LATE_FRAC))    # e.g., 0.60 * T
    CHECK_STEPS = list(range(start, end, max(1, STEP_STRIDE)))
    FREEZE_AFTER_STEP = end                      # from this step onward: freeze to one latent
    frozen_idx = None                            # set once when we first hit freeze region

    print(f"[FK] checkpoints (0-based): {CHECK_STEPS}")
    print(f"[FK] freeze from step >= {FREEZE_AFTER_STEP}")
    print(f"[FK] N={N_PARTICLES}  beta={BETA_CONST}  ESS_th={ESS_THRESHOLD}  mode={POTENTIAL_MODE}")

    # Cosmos expects guidance sequences even if unused → send zeros
    guidance_step = [0] * NUM_STEPS
    guidance_lr   = [0.0] * NUM_STEPS

    # --- 4) FK/SMC callback
    @torch.inference_mode()
    def fk_callback(pipe_obj, step: int, timestep: int, callback_kwargs: dict, **_):
        nonlocal last_phi, running_sum, running_max, frozen_idx, weights, product_of_potentials, population_rs

        latents = callback_kwargs.get("latents", None)  # (B, F, C, H, W)

        # pipe_obj.scheduler.config.stochastic_sampling = True

        # --------- FREEZE REGION: replicate the chosen particle; no more scoring/resampling ---------
        if step >= FREEZE_AFTER_STEP:
            if frozen_idx is None:
                # Choose once at the boundary (cheap): best by current weights
                # frozen_idx = int(weights.argmax().item())

                # If you prefer a one-time precise pick (slower), uncomment:
                vids_full = decode_full(pipe_obj, latents)
                surprise_t = vjepa_surprise_batch(vids_full, encoder, target_encoder, predictor)
                frozen_idx = int(surprise_t.argmin().item())

                print(f"[FK] FREEZE at step {step}: locking particle {frozen_idx}")

            idx = torch.full((latents.shape[0],), frozen_idx, device=latents.device, dtype=torch.long)
            latents = latents.index_select(0, idx)

            # Keep weights uniform so downstream logic (if any) won't try to resample
            weights.fill_(1.0 / weights.numel())

            # Turn off stochastic sampling for the rest of the trajectory
            # pipe_obj.scheduler.config.stochastic_sampling = False

            return {"latents": latents}
        # --------------------------------------------------------------------------------------------

        if step in CHECK_STEPS:
            # 1) Decode current latents
            vids_full = decode_full(pipe_obj, latents)  # (B,C,T,H,W) in [-1,1]

            # 2) Score -> phi_t (higher is better). Using 1 - surprise keeps sign consistent; -surprise also works.
            surprise_t = vjepa_surprise_batch(vids_full, encoder, target_encoder, predictor)  # (B,)
            phi_t = 1.0 - surprise_t

            # 3) Potential selection
            if POTENTIAL_MODE == "diff":
                phi = phi_t - last_phi
            elif POTENTIAL_MODE == "sum":
                running_sum = running_sum + phi_t
                phi = running_sum
            elif POTENTIAL_MODE == "max":
                running_max = torch.maximum(running_max, phi_t)
                phi = running_max
            else:
                raise ValueError(f"Unknown POTENTIAL_MODE: {POTENTIAL_MODE}")

            # Maintain product of potentials (un-normalized) for final-step correction
            pot_term = torch.exp((BETA_CONST * phi).clamp(min=-60.0, max=60.0))
            product_of_potentials = product_of_potentials * pot_term
            population_rs = phi

            # If this is the final point, use corrected weight BEFORE any resampling
            last_check_step = CHECK_STEPS[-1] if len(CHECK_STEPS) > 0 else -1
            is_final_point = (step == last_check_step) or (step == (NUM_STEPS - 1))
            if is_final_point:
                w = torch.exp((BETA_CONST * population_rs).clamp(min=-60.0, max=60.0)) / (product_of_potentials + 1e-8)
                w = torch.clamp(w, 0.0, 1e10)
                w[w.isnan()] = 0.0

                normalized_w = w / (w.sum() + 1e-8)
                ess = 1.0 / (normalized_w.pow(2).sum() + 1e-8)
                print(f"      [final-correction] ESS={float(ess):.3f}")

                if ess < 0.5 * N_PARTICLES:
                    print(f"      [final-correction] RESAMPLE at step {step} with ESS={float(ess):.3f}")
                    idx = torch.multinomial(w, num_samples=w.numel(), replacement=True)
                    latents = latents.index_select(0, idx)
                    weights.fill_(1.0 / w.numel())
                    last_phi = last_phi.index_select(0, idx) if POTENTIAL_MODE == "diff" else last_phi
                    running_sum = running_sum.index_select(0, idx)
                    running_max = running_max.index_select(0, idx)
                    product_of_potentials = product_of_potentials.index_select(0, idx)
                    population_rs = population_rs.index_select(0, idx)
                    lineage.copy_(lineage.index_select(0, idx))
                else:
                    weights.copy_(normalized_w)

                return {"latents": latents}

            # 4) Weights update (non-final steps)
            z = BETA_CONST * phi
            z = z - z.max()
            incr = torch.exp(z.clamp(min=-60.0))
            new_w = weights * incr
            new_w = new_w / (new_w.sum() + 1e-8)

            # 5) Metrics
            ess_over_n = ess_normed(new_w) / new_w.numel()
            ent_bits   = weight_entropy_bits(new_w)
            best_loss, best_idx = float(surprise_t.min()), int(surprise_t.argmin())
            worst_loss          = float(surprise_t.max())
            mean_loss           = float(surprise_t.mean())
            std_loss            = float(surprise_t.std(unbiased=False))

            do_resample = (ess_over_n < ESS_THRESHOLD)  # ESS-only

            # Debug
            print(
                f"[FK] step={step:02d} (t={int(timestep)}), "
                f"loss min/mean±std/max= {best_loss:.4f} / {mean_loss:.4f}±{std_loss:.4f} / {worst_loss:.4f}, "
                f"beta={BETA_CONST:.1f}, ESS/N={ess_over_n:.3f}, H(bits)={ent_bits:.2f}, resample={do_resample}"
            )
            if DEBUG_TOPK > 0:
                topk_phi_t = torch.topk(phi_t, k=min(DEBUG_TOPK, phi_t.numel()))
                print(f"      top{topk_phi_t.values.numel()} φ_t idx={topk_phi_t.indices.tolist()} "
                      f"vals={[f'{v:.4f}' for v in topk_phi_t.values.tolist()]} (best_loss idx={best_idx})")
                topk_w = torch.topk(new_w, k=min(DEBUG_TOPK, new_w.numel()))
                print(f"      top{topk_w.values.numel()} w   idx={topk_w.indices.tolist()} "
                      f"vals={[f'{v:.3f}' for v in topk_w.values.tolist()]}")

            if PRINT_WEIGHTS:
                print(f"      w = {[float(x) for x in new_w.tolist()]}")

            # 6) Resampling
            if do_resample:
                idx = torch.multinomial(new_w, num_samples=new_w.numel(), replacement=True)
                latents = latents.index_select(0, idx)

                # reset weights, carry state for selected particles
                weights.fill_(1.0 / new_w.numel())
                last_phi = phi_t.index_select(0, idx) if POTENTIAL_MODE == "diff" else last_phi
                running_sum = running_sum.index_select(0, idx)
                running_max = running_max.index_select(0, idx)
                product_of_potentials = product_of_potentials.index_select(0, idx)
                population_rs = population_rs.index_select(0, idx)
                lineage.copy_(lineage.index_select(0, idx))

                # Debug lineage histogram
                print(f"RESAMPLE! idx[:{PRINT_IDX_HEAD}]={idx[:PRINT_IDX_HEAD].tolist()}  -> weights reset")
                counts = torch.bincount(lineage, minlength=N_PARTICLES).cpu().numpy()
                head_bins = min(LINEAGE_HIST_MAX, len(counts))
                hist_str = ", ".join([f"{i}:{int(c)}" for i, c in enumerate(counts[:head_bins]) if c > 0])
                tail_nonzero = [(i, int(c)) for i, c in enumerate(counts[head_bins:]) if c > 0]
                if hist_str or tail_nonzero:
                    print(f"      lineage copies (orig_seed_id:count) head -> {hist_str}" +
                          (f" ... +{len(tail_nonzero)} more" if tail_nonzero else ""))
            else:
                # No resample: persist weights; roll last_phi for diff mode
                weights.copy_(new_w)
                if POTENTIAL_MODE == "diff":
                    last_phi = phi_t
                print(f"      no resample (ESS/N={ess_over_n:.3f} ≥ {ESS_THRESHOLD:.3f})")

            return {"latents": latents}

        return {}  # no-op outside CHECK_STEPS

    # --- 5) Run steered sampling (neutral internal guidance)
    out = pipe(
        image=[init_image] * N_PARTICLES,
        prompt=[PROMPT] * N_PARTICLES,
        negative_prompt=[NEG_PROMPT] * N_PARTICLES,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=gens,
        loss_fn="slice_pred",
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        guidance_frequency=1,
        fps=FPS,
        travel_time=(0, 0),
        callback_on_step_end=fk_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    # --- 6) Final selection by V-JEPA
    vids_btchw_list = []
    for particle_frames in out.frames:
        frames_t = []
        for pil_frame in particle_frames:
            arr = np.array(pil_frame)
            ft  = torch.from_numpy(arr).float().permute(2, 0, 1)  # (C,H,W)
            ft  = (ft / 255.0) * 2.0 - 1.0
            frames_t.append(ft)
        particle_video = torch.stack(frames_t, dim=0).permute(1, 0, 2, 3)  # (C,T,H,W)
        vids_btchw_list.append(particle_video)

    vids_btchw = torch.stack(vids_btchw_list, dim=0).cuda()  # (B,C,T,H,W)
    final_surprise = vjepa_surprise_batch(vids_btchw, encoder, target_encoder, predictor)
    best_idx = int(final_surprise.argmin().item())
    print(f"[FK] FINAL best particle: {best_idx}  V-JEPA loss: {float(final_surprise[best_idx]):.6f}")

    export_to_video(out.frames[best_idx], "output_fk_vjepa_cosmos1.mp4", fps=FPS)
    print("Saved: output_fk_vjepa_cosmos1.mp4")

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    main()
