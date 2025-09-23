#!/usr/bin/env python3
# unified_search_vjepa_cogvideox.py
# One callback to rule them all: SVDD and DSearch are just configs.

import os
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from PIL import Image

from pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video, load_image

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
SEED0          = 42

# ===================== unified search config ======================
@dataclass
class SearchCfg:
    # Topology
    num_beams: int = 1          # B
    branch_K: int = 4           # K children per beam
    keep_beams: int = 1         # how many beams to keep after per-beam selection (<= num_beams)
    # Scoring / selection
    accumulate: bool = False    # cumulative over steps (DSearch) vs instantaneous (SVDD-like)
    select_softmax: bool = True # per-beam: pick child by softmax(beta*obj) if True; else argmax
    beta: float = 10.0          # temperature for softmax (ignored if select_softmax=False)
    # Schedule
    stride: int = 5             # evaluate every k steps
    start_frac: float = 0.0     # start of search window [0,1]
    end_frac: float = 1.0       # end of search window [0,1]
    freeze_tail: bool = False   # after end_frac: collapse to best global child and finish with batch=1
    # Reward compute
    eval_decode_stride: int = 2 # temporal stride during reward
    print_topk: int = 4

# Presets (no branching; just choose a cfg)
def svdd_preset(K=4, beta=10.0, stride=5) -> SearchCfg:
    return SearchCfg(
        num_beams=1, branch_K=K, keep_beams=1,
        accumulate=False, select_softmax=True, beta=beta,
        stride=stride, start_frac=0.0, end_frac=1.0,
        freeze_tail=False, eval_decode_stride=2, print_topk=4
    )

def dsearch_preset(B=3, K=2, stride=5) -> SearchCfg:
    return SearchCfg(
        num_beams=B, branch_K=K, keep_beams=B,
        accumulate=True, select_softmax=False, beta=0.0,
        stride=stride, start_frac=0.10, end_frac=0.90,
        freeze_tail=True, eval_decode_stride=2, print_topk=4
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
    if -0.05 <= xmin and xmax <= 1.05:
        return x * 2.0 - 1.0
    if 0.0 <= xmin and xmax <= 255.0:
        return (x / 127.5) - 1.0
    return x

@torch.inference_mode()
def decode_full(pipe, latents):
    latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    frames = pipe.decode_latents(latents)
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
        raise RuntimeError(f"Unexpected decoded shape/type: {type(frames)} {getattr(frames,'shape',None)}")
    frames = ensure_bcthw(frames)
    frames = to_minus1_1(frames)
    return frames

@torch.inference_mode()
def vjepa_surprise_batch(vids_btchw: torch.Tensor,
                         encoder, target_encoder, predictor,
                         img_size=256, window=16, stride=8, context=8,
                         masking_mode="causal", mask_ratio=0.75, loss_mode="max") -> torch.Tensor:
    vids_btchw = ensure_bcthw(vids_btchw).to(dtype=torch.float32)
    B = vids_btchw.shape[0]
    out = torch.empty(B, device=vids_btchw.device, dtype=torch.float32)
    for i in range(B):
        loss = compute_vjepa_loss_sliding_window(
            video_tensor=vids_btchw[i:i+1],
            encoder=encoder, target_encoder=target_encoder, predictor=predictor,
            img_size=img_size, window_size=window, stride=stride,
            masking_mode=masking_mode, context_frames=context, mask_ratio=mask_ratio,
            spatial_pred_mask_scale=None, temporal_pred_mask_scale=None,
            aspect_ratio=None, npred=None, max_context_frames_ratio=None,
            loss_exp=2, is_vae_output=True, seed=42, mode=loss_mode,
        )
        out[i] = float(loss)
    return out  # (B,)

# ===================== unified callback ======================
def make_unified_callback(cfg: SearchCfg, encoder, target_encoder, predictor):
    state = {"cumulative": None, "frozen": False, "check_steps": None, "freeze_from": None, "active_beams": None}

    def _maybe_init_schedule(pipe_obj):
        if state["check_steps"] is not None:
            return
        # NUM_STEPS is provided externally; we still compute using cfg and trust the loop's step index
        total = NUM_STEPS
        start = int(round(total * cfg.start_frac))
        end   = int(round(total * cfg.end_frac))
        state["check_steps"] = list(range(start, end, max(1, cfg.stride)))
        state["freeze_from"] = end
        print(f"[UNIFIED] checkpoints: {state['check_steps']}  freeze_from={state['freeze_from']}, "
              f"B={cfg.num_beams}, K={cfg.branch_K}, keep={cfg.keep_beams}, "
              f"accumulate={cfg.accumulate}, softmax={cfg.select_softmax}, beta={cfg.beta}")

    @torch.inference_mode()
    def cb(pipe_obj, step: int, timestep: int, callback_kwargs: dict, **_):
        _maybe_init_schedule(pipe_obj)
        latents = callback_kwargs.get("latents", None)
        if latents is None:
            raise RuntimeError("Pipeline must expose 'latents' via callback_on_step_end_tensor_inputs=['latents'].")

        K = cfg.branch_K
        # Initialize active beams on first call
        if state["active_beams"] is None:
            state["active_beams"] = cfg.num_beams
        
        current_B = state["active_beams"]
        expected_batch = current_B * K
        assert latents.shape[0] == expected_batch or state["frozen"], \
            f"Expected batch {expected_batch}, got {latents.shape[0]} (unless frozen)."

        # lazy init cumulative
        if state["cumulative"] is None or state["cumulative"].shape[0] != latents.shape[0]:
            state["cumulative"] = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)

        # Freeze tail (optional; works for any config)
        if (cfg.freeze_tail) and (step >= state["freeze_from"]) and not state["frozen"]:
            vids = decode_full(pipe_obj, latents)
            if cfg.eval_decode_stride > 1:
                vids = vids[:, :, ::cfg.eval_decode_stride, :, :]
            surprise = vjepa_surprise_batch(vids, encoder, target_encoder, predictor)
            reward = 1.0 - surprise
            if cfg.accumulate:
                state["cumulative"] = state["cumulative"] + reward
                best_idx = int(torch.argmax(state["cumulative"]).item())
            else:
                # In instantaneous mode, pick the best current reward
                best_idx = int(torch.argmax(reward).item())
            print(f"[UNIFIED] FREEZE step={step}: best={best_idx} cum={float(state['cumulative'][best_idx]) if cfg.accumulate else float(reward[best_idx]):.4f}")
            new_latents = latents[best_idx:best_idx+1]
            state["frozen"] = True
            return {"latents": new_latents}

        if (state["check_steps"] is None) or (step not in state["check_steps"]) or state["frozen"]:
            return {}

        # 1) decode + reward
        vids = decode_full(pipe_obj, latents)
        if cfg.eval_decode_stride > 1:
            vids = vids[:, :, ::cfg.eval_decode_stride, :, :]
        surprise = vjepa_surprise_batch(vids, encoder, target_encoder, predictor)
        step_reward = 1.0 - surprise

        # 2) update objective
        obj = state["cumulative"] + step_reward if cfg.accumulate else step_reward

        # 3) per-beam fusion/selection
        obj_g = obj.view(current_B, K)  # (current_B, K)
        if (not cfg.accumulate) and cfg.select_softmax and cfg.beta > 0:
            # SVDD-style soft mix
            z = cfg.beta * obj_g
            z = z - z.max(dim=1, keepdim=True).values
            w = torch.softmax(torch.clamp(z, min=-60.0), dim=1)  # (current_B, K)
            # Weighted fuse latents per beam
            new_latents_list = []
            for b in range(w.shape[0]):
                wb = w[b].view(K, *([1] * (latents.ndim - 1)))           # (K,1,1,1,1)
                lb = latents[b*K:(b+1)*K]                                 # (K, C, T, H, W)
                fused = torch.sum(wb * lb, dim=0, keepdim=True)           # (1, C, T, H, W)
                new_latents_list.append(fused)
            new_latents = torch.cat(new_latents_list, dim=0)              # (current_B, ...)
            # Repeat each fused beam K times to keep batch = current_B*K
            new_latents = new_latents.repeat_interleave(K, dim=0)         # (current_B*K, ...)
            # Reset cumulative for instantaneous mode
            state["cumulative"] = torch.zeros(new_latents.shape[0], device=new_latents.device, dtype=torch.float32)
            # No change in active beam count for SVDD
            return {"latents": new_latents}
        else:
            child_idx = torch.argmax(obj_g, dim=1)  # (current_B,)

        base = torch.arange(obj_g.shape[0], device=obj.device) * K
        best_abs = base + child_idx  # (current_B,)

        # 4) keep top 'keep_beams' beams by their chosen-child score
        best_vals = obj.index_select(0, best_abs)
        keep = min(cfg.keep_beams, best_vals.numel())
        order = torch.topk(best_vals, k=keep, largest=True).indices
        chosen_abs = best_abs.index_select(0, order)  # (keep,)

        # 5) replicate each kept child K times to form next batch
        chosen_latents = latents.index_select(0, chosen_abs)          # (keep, ...)
        new_latents = chosen_latents.repeat_interleave(K, dim=0)      # (keep*K, ...)

        # 6) propagate cumulative in lockstep with new batch
        if cfg.accumulate:
            kept_vals = best_vals.index_select(0, order)              # (keep,)
            state["cumulative"] = kept_vals.repeat_interleave(K, dim=0).contiguous()
        else:
            # reset cumulative (instantaneous mode): keep zeros so next obj = step_reward
            state["cumulative"] = torch.zeros_like(new_latents[:, 0, 0, 0, 0], dtype=torch.float32, device=new_latents.device)
        
        # 7) update active beam count
        state["active_beams"] = keep

        # Logs
        if cfg.print_topk > 0:
            tkv, tki = torch.topk(best_vals, k=min(cfg.print_topk, best_vals.numel()))
            print(f"[UNIFIED] step={step:02d} t={int(timestep)} "
                  f"surprise[min/mean/max]={float(surprise.min()):.4f}/"
                  f"{float(surprise.mean()):.4f}/"
                  f"{float(surprise.max()):.4f}  "
                  f"keep={keep}  top-beams idx={tki.tolist()} vals={[f'{v:.4f}' for v in tkv.tolist()]}")

        return {"latents": new_latents}

    return cb

# ===================== main ======================
def main():
    # 1) Pipeline
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe.to("cuda")
    if hasattr(pipe, "vae"):
        try:
            pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
        except Exception:
            pass

    init_image = load_image(INIT_IMAGE).convert("RGB")

    # 2) V-JEPA models
    encoder, target_encoder, predictor, _ = load_vjepa_models_torchhub("vit_giant")
    encoder.eval().cuda(); target_encoder.eval().cuda(); predictor.eval().cuda()

    # ======= Choose ONE config (no branching logic needed) =======
    # A) SVDD behavior
    cfg = svdd_preset(K=4, beta=10.0, stride=5)

    # B) DSearch behavior
    # cfg = dsearch_preset(B=3, K=2, stride=5)

    # Prepare
    global NUM_STEPS
    NUM_STEPS = NUM_STEPS  # already defined at top; kept for clarity
    N = cfg.num_beams * cfg.branch_K
    gens = [torch.Generator(device="cuda").manual_seed(SEED0 + i) for i in range(N)]

    # 3) Unified callback
    cb = make_unified_callback(cfg, encoder, target_encoder, predictor)

    # 4) Run once with unified search
    out = pipe(
        video=[init_image] * N,
        prompt=[PROMPT] * N,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        use_dynamic_cfg=True,
        num_videos_per_prompt=1,
        eta=0.0,  # Deterministic DDIM for reproducible beam search
        generator=gens,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
        # Keep your pacing from SMC
        guidance_step=build_seq("0x50", NUM_STEPS, is_float=False),
        guidance_lr=build_seq("0x50", NUM_STEPS, is_float=True),
        guidance_frequency=1,
        additional_inputs=None,
        travel_time=(-1, -1),
    )

    # 5) Final pick via V-JEPA
    if not out.frames:
        raise RuntimeError("No frames returned from inference!")

    vids_btchw_list = []
    for i, particle_frames in enumerate(out.frames):
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

    vids_btchw = torch.stack(vids_btchw_list, dim=0).cuda()
    final_surprise = vjepa_surprise_batch(vids_btchw, encoder, target_encoder, predictor)
    best_idx = int(final_surprise.argmin().item())
    print(f"[UNIFIED] FINAL best particle: {best_idx}  V-JEPA loss: {float(final_surprise[best_idx]):.6f}")

    out_path = "output_unified_vjepa.mp4"
    export_to_video(out.frames[best_idx], out_path, fps=FPS)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    main()
