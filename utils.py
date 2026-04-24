# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s

import torch
import torch.nn as nn
import torch.nn.functional as F
import decord
from decord import VideoReader
import numpy as np
import sys
import os
import copy
from torchvision import transforms
from diffusers.utils import export_to_video
from PIL import Image
from einops import rearrange

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vjepa2"))
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope, vit_huge_rope
from src.models.predictor import vit_predictor
from src.models.ac_predictor import vit_ac_predictor
from src.masks.utils import apply_masks

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def set_deterministic(seed=42):
    """Set deterministic behavior for reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def _clean_backbone_key(state_dict):
    for key, val in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        state_dict[key] = val
    return state_dict


def log_grad_spread(g, delta_base, step_i, sample_idx: int = 0, topk: int = 5):
    """
    g            : [B,C,T,H,W]   (∂L/∂x_t)
    delta_base   : [B,C,T,H,W]   (vanilla solver step per frame)
    step_i       : int           (k in your loop)
    sample_idx   : which batch element to print
    topk         : how many top frames to summarize
    """
    print('g', g.shape)
    print('delta_base', delta_base.shape)
    eps = 1e-12
    assert g.dim() == 5 and delta_base.shape == g.shape, "shape mismatch"
    B, C, T, H, W = g.shape
    b = min(sample_idx, B-1)

    # Per-frame L2 norms over C,H,W
    reduce_chw = (1, 3, 4)
    g_t = g.pow(2).sum(dim=reduce_chw).sqrt()                   # [B,T]
    d_t = delta_base.pow(2).sum(dim=reduce_chw).sqrt()          # [B,T]
    dot_t = (g * delta_base).sum(dim=reduce_chw)                # [B,T]
    cos_t = dot_t / (g_t * d_t + eps)                           # [B,T] per-frame cosine

    # Normalized energy distribution over frames
    p_t = g_t / (g_t.sum(dim=1, keepdim=True) + eps)            # [B,T], sum_t p_t = 1

    # Concentration metrics
    hhi = (p_t**2).sum(dim=1)                                   # Herfindahl index
    eff_frames = 1.0 / (hhi + eps)                              # "effective number of frames"

    # How many frames cover 50% / 90% of the mass?
    p_sorted, idx_sorted = torch.sort(p_t[b], descending=True)
    csum = torch.cumsum(p_sorted, dim=0)
    k50 = int((csum >= 0.50).nonzero(as_tuple=False)[0]) + 1
    k90 = int((csum >= 0.90).nonzero(as_tuple=False)[0]) + 1

    # Top-k summary
    K = min(topk, T)
    top_idx = idx_sorted[:K]
    top_mass = p_sorted[:K].sum().item()
    top_cos_mean = cos_t[b, top_idx].mean().item()

    # Compact, readable printout
    tops = [(int(i), float(p_t[b, i]), float(cos_t[b, i])) for i in top_idx]
    print(f"[k={step_i}] grad spread (b={b}): eff_frames={eff_frames[b].item():.2f}, "
        f"k50={k50}, k90={k90}, top{K}_mass={top_mass:.2f}, top{K}_cos={top_cos_mean:.3f}")
    print(f"          top{K} frames (idx, p_t, cos): {tops}")

    # Optional: an ASCII bar for p_t (one character per frame, scaled)
    width = 30
    bars = ''.join('█' * max(1, int(width * float(p))) for p in p_t[b].tolist())
    print(f"          p_t bars (T={T}): {bars}")

def build_pt_video_transform(img_size):
    """Build video preprocessing transform."""
    eval_transform = video_transforms.Compose([
        video_transforms.Resize((img_size, img_size), interpolation="bilinear"),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return eval_transform

def get_video(path, max_frames=49):
    """Load and sample video frames."""
    vr = VideoReader(path)
    num_frames = len(vr)
    frame_count = min(max_frames, num_frames)
    # Uniformly sample frame indices
    frame_idx = np.linspace(0, num_frames - 1, frame_count, dtype=int)
    video = vr.get_batch(frame_idx).asnumpy()
    return video

def create_repeated_frame_video(source_video_path, num_frames, output_path):
    """Create a video with the last frame repeated num_frames times."""
    if os.path.exists(output_path):
        return  # Already exists, skip creation

    # Load source video and get last frame
    source_video = get_video(source_video_path, max_frames=5)
    last_frame = source_video[-1]  # [H, W, C], RGB format, values 0-255

    # Ensure values are in uint8 range [0, 255]
    if last_frame.dtype != np.uint8:
        last_frame = np.clip(last_frame, 0, 255).astype(np.uint8)

    # Convert numpy array to PIL Image
    last_frame_pil = Image.fromarray(last_frame)

    # Create repeated frames: list of PIL Images
    repeated_frames = [last_frame_pil.copy() for _ in range(num_frames)]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use diffusers export_to_video function
    export_to_video(repeated_frames, output_path, fps=16)
    print(f"✅ Created repeated frame video: {output_path} ({num_frames} frames)")

def build_dinov2_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

def load_dinov2_model():
    backbone_model  = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitl14_reg")
    backbone_model.eval()
    backbone_model.cuda()
    return backbone_model

def load_vjepa_model_source(model, num_frames=64):
    """Load V-JEPA model with weights."""
    CHECKPOINT_DIR = os.environ.get("VJEPA_CHECKPOINT_DIR", "./checkpoints")
    img_size = 384 if "384" in model else 256
    if model == "vith" or model == "vit_huge":
        encoder = vit_huge_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = os.path.join(CHECKPOINT_DIR, "vith.pt")

    elif model == "vitg" or model == "vit_giant":
        encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = os.path.join(CHECKPOINT_DIR, "vitg.pt")
    elif model == "vitg384" or model == "vit_giant_384":
        encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = os.path.join(CHECKPOINT_DIR, "vitg-384.pt")
    elif model == "vitgac" or model == "vit_giant_ac":
        encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = os.path.join(CHECKPOINT_DIR, "vjepa2-ac-vitg.pt")
    else:
        raise ValueError(f"Unknown model: {model}. Use 'vith', 'vitg' or 'vitg384'.")


    encoder.cuda().eval()
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict_cleaned = _clean_backbone_key(state_dict["encoder"])
    encoder.load_state_dict(state_dict_cleaned, strict=True)

    target_encoder = copy.deepcopy(encoder)


    predictor = load_vjepa_predictor(model_path, encoder, img_size)
    return encoder, target_encoder, predictor, img_size

def load_vjepa_predictor(model_path, encoder, img_size=256):
    """Load V-JEPA predictor with weights that match the encoder."""
    model = vit_predictor(
        img_size=(img_size, img_size),
        patch_size=encoder.patch_size,
        use_mask_tokens=True,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        num_frames=encoder.num_frames,
        tubelet_size=encoder.tubelet_size,
        depth=12,
        num_heads=12,
        num_mask_tokens=10,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
    )
    model.cuda().eval()
    state_dict = torch.load(model_path, map_location="cpu")
    predictor_state_dict = _clean_backbone_key(state_dict["predictor"])
    model.load_state_dict(predictor_state_dict, strict=True)
    return model


def load_vjepa_models_torchhub(model):
    """
    Load V-JEPA models for loss computation.

    Args:
        model_path (str): Path to the V-JEPA model checkpoint
        img_size (int): Image size for processing

    Returns:
        tuple: (encoder, target_encoder, predictor) models
    """
    img_size = 384 if "384" in model else 256
    if model == 'vith' or model == 'vit_huge':
        encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
    elif model == 'vitg' or model == 'vit_giant':
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")
    elif model == 'vitg384' or model == 'vit_giant_384':
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant_384")
    elif model == 'vitgac' or model == 'vit_giant_ac':
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    else:
        raise ValueError(f"Unknown model: {model}. Use 'vith' or 'vjepa2'.")

    target_encoder = copy.deepcopy(encoder)


    return encoder, target_encoder, predictor, img_size

def generate_vjepa_masks(masking_mode, batch_size, img_size, frames_per_clip, encoder,
                        context_frames=15, mask_ratio=0.75, device="cuda",
                        spatial_pred_mask_scale=(0.2, 0.8), temporal_pred_mask_scale=(1.0, 1.0),
                        aspect_ratio=(0.3, 3.0), npred=1, max_context_frames_ratio=1.0,
                        seed=42, window_start=0, total_frames=None):
    """
    Generate masks for V-JEPA loss computation using the actual training masking strategy.

    Args:
        masking_mode (str): "block" for V-JEPA block masking, "causal" for temporal masking, "expanding_causal" for expanding context window, or "random" for random token masking
        batch_size (int): Batch size
        img_size (int): Image size for processing
        frames_per_clip (int): Total frames in clip
        encoder: V-JEPA encoder model (used to get patch_size and tubelet_size)
        context_frames (int): Number of context frames (only used for causal mode)
        mask_ratio (float): Ratio of tokens to mask (only used for random mode)
        device: Device to create tensors on
        spatial_pred_mask_scale (tuple): (min, max) spatial scale for prediction blocks
        temporal_pred_mask_scale (tuple): (min, max) temporal scale for prediction blocks
        aspect_ratio (tuple): (min, max) aspect ratio range for blocks
        npred (int): Number of prediction blocks to sample
        max_context_frames_ratio (float): Maximum fraction of frames that can be context
        seed (int): Random seed for reproducible masking
        window_start (int): Starting frame position of current window (used for expanding_causal mode)
        total_frames (int): Total number of frames in the full sequence (used for expanding_causal mode)

    Returns:
        tuple: (ctxt_positions, tgt_positions) - masks for context and target tokens
    """
    grid_size = img_size // encoder.patch_size  # spatial grid size (H, W in patches)
    grid_depth = frames_per_clip // encoder.tubelet_size  # temporal grid size (T in tubelets)
    total_tokens = int(grid_size**2 * grid_depth)

    if masking_mode == "block":
        # V-JEPA block-based masking strategy
        return _generate_block_masks(
            batch_size=batch_size,
            height=grid_size,
            width=grid_size,
            duration=grid_depth,
            spatial_pred_mask_scale=spatial_pred_mask_scale,
            temporal_pred_mask_scale=temporal_pred_mask_scale,
            aspect_ratio=aspect_ratio,
            npred=npred,
            max_context_frames_ratio=max_context_frames_ratio,
            device=device,
            seed=seed
        )
    elif masking_mode == "causal":
        # Causal masking: use first frames as context, predict future frames
        context_depth = context_frames // encoder.tubelet_size
        future_steps = grid_depth - context_depth

        # Validate that we have reasonable splits
        if future_steps <= 0:
            raise ValueError(f"Context frames ({context_frames}) too large for frames_per_clip ({frames_per_clip})")

        N_context = int(grid_size**2 * context_depth)
        N_pred = int(grid_size**2 * future_steps)

        # Create position masks - these are token indices, not frame indices
        ctxt_positions = torch.arange(N_context, device=device).unsqueeze(0).repeat(batch_size, 1)
        tgt_positions = torch.arange(N_pred, device=device).unsqueeze(0).repeat(batch_size, 1)
        tgt_positions += N_context  # Offset by context size

    elif masking_mode == "expanding_causal":
        # Expanding causal masking: use all frames from beginning up to current window as context
        if total_frames is None:
            raise ValueError("total_frames must be provided for expanding_causal mode")

        # Calculate how many frames from the beginning to use as context
        # This includes all frames from start (0) up to the current window start + some frames within window
        context_frames_total = window_start + context_frames
        context_frames_total = min(context_frames_total, total_frames)  # Don't exceed total frames

        # Convert frame counts to token depths
        context_depth_total = context_frames_total // encoder.tubelet_size
        current_window_depth = frames_per_clip // encoder.tubelet_size

        # Predict the remaining frames in current window
        prediction_depth = current_window_depth - (context_frames // encoder.tubelet_size)
        prediction_depth = max(1, prediction_depth)  # Ensure we have at least 1 frame to predict

        # Calculate token counts
        N_context = int(grid_size**2 * context_depth_total)
        N_pred = int(grid_size**2 * prediction_depth)

        # For expanding context, we need to map tokens correctly:
        # Context tokens span from beginning of sequence to current window position
        # Target tokens are the remaining frames in the current window

        # Context includes tokens from start to the context portion of current window
        ctxt_positions = torch.arange(N_context, device=device).unsqueeze(0).repeat(batch_size, 1)

        # Target tokens are the prediction portion of current window
        # They start after the context portion of the current window
        context_in_window = (context_frames // encoder.tubelet_size) * grid_size**2
        window_start_token = (window_start // encoder.tubelet_size) * grid_size**2
        tgt_start = window_start_token + context_in_window
        tgt_positions = torch.arange(tgt_start, tgt_start + N_pred, device=device).unsqueeze(0).repeat(batch_size, 1)

    elif masking_mode == "random":
        # Random masking: randomly select tokens to mask
        num_mask = int(total_tokens * mask_ratio)
        num_keep = total_tokens - num_mask

        # Create random permutations for each batch item
        batch_keep_masks = []
        batch_pred_masks = []

        for b in range(batch_size):
            # Random permutation of all token indices
            perm = torch.randperm(total_tokens, device=device)

            # Split into keep (context) and mask (predict) tokens
            keep_indices = perm[:num_keep].sort()[0]  # Sort to maintain some order
            mask_indices = perm[num_keep:].sort()[0]  # Sort to maintain some order

            batch_keep_masks.append(keep_indices.unsqueeze(0))  # [1, num_keep]
            batch_pred_masks.append(mask_indices.unsqueeze(0))  # [1, num_mask]

        # Stack all batch items
        ctxt_positions = torch.cat(batch_keep_masks, dim=0)  # [B, num_keep]
        tgt_positions = torch.cat(batch_pred_masks, dim=0)   # [B, num_mask]

    else:
        raise ValueError(f"Unknown masking_mode: {masking_mode}. Use 'block', 'causal', 'expanding_causal', or 'random'.")

    return ctxt_positions, tgt_positions


def _sample_block_size(generator, duration, height, width, temporal_scale, spatial_scale, aspect_ratio_scale):
    """
    Sample block size for V-JEPA masking following the training implementation.

    Args:
        generator: PyTorch random generator
        duration (int): Number of temporal patches
        height (int): Number of spatial patches (height)
        width (int): Number of spatial patches (width)
        temporal_scale (tuple): (min, max) temporal scale
        spatial_scale (tuple): (min, max) spatial scale
        aspect_ratio_scale (tuple): (min, max) aspect ratio

    Returns:
        tuple: (t, h, w) block dimensions
    """
    import math

    # Sample temporal block mask scale
    _rand = torch.rand(1, generator=generator).item()
    min_t, max_t = temporal_scale
    temporal_mask_scale = min_t + _rand * (max_t - min_t)
    t = max(1, int(duration * temporal_mask_scale))

    # Sample spatial block mask scale
    _rand = torch.rand(1, generator=generator).item()
    min_s, max_s = spatial_scale
    spatial_mask_scale = min_s + _rand * (max_s - min_s)
    spatial_num_keep = int(height * width * spatial_mask_scale)

    # Sample block aspect-ratio
    _rand = torch.rand(1, generator=generator).item()
    min_ar, max_ar = aspect_ratio_scale
    aspect_ratio = min_ar + _rand * (max_ar - min_ar)

    # Compute block height and width (given scale and aspect-ratio)
    h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
    w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
    h = min(h, height)
    w = min(w, width)

    return (t, h, w)


def _sample_block_mask(b_size, duration, height, width, max_context_duration):
    """
    Sample a block mask for V-JEPA masking following the training implementation.

    Args:
        b_size (tuple): (t, h, w) block dimensions
        duration (int): Total temporal patches
        height (int): Total spatial patches (height)
        width (int): Total spatial patches (width)
        max_context_duration (int): Maximum context duration

    Returns:
        torch.Tensor: 3D mask of shape (duration, height, width)
    """
    t, h, w = b_size
    top = torch.randint(0, height - h + 1, (1,))
    left = torch.randint(0, width - w + 1, (1,))
    start = torch.randint(0, duration - t + 1, (1,))

    mask = torch.ones((duration, height, width), dtype=torch.int32)
    mask[start : start + t, top : top + h, left : left + w] = 0

    # Context mask will only span the first X frames
    if max_context_duration < duration:
        mask[max_context_duration :, :, :] = 0

    return mask


def _generate_block_masks(batch_size, height, width, duration, spatial_pred_mask_scale,
                         temporal_pred_mask_scale, aspect_ratio, npred, max_context_frames_ratio,
                         device, seed):
    """
    Generate V-JEPA block masks following the actual training implementation.

    This replicates the behavior of _MaskGenerator.__call__() from multiseq_multiblock3d.py
    """
    max_context_duration = max(1, int(duration * max_context_frames_ratio))

    # Set up generator with seed for reproducible block sizes
    g = torch.Generator()
    # g.manual_seed(seed)

    # Sample prediction block size using seed (same for all batch items)
    p_size = _sample_block_size(
        generator=g,
        duration=duration,
        height=height,
        width=width,
        temporal_scale=temporal_pred_mask_scale,
        spatial_scale=spatial_pred_mask_scale,
        aspect_ratio_scale=aspect_ratio,
    )

    collated_masks_pred, collated_masks_enc = [], []
    min_keep_enc = min_keep_pred = duration * height * width

    for _ in range(batch_size):
        empty_context = True
        while empty_context:
            # Start with all tokens available
            mask_e = torch.ones((duration, height, width), dtype=torch.int32)

            # Apply npred prediction blocks
            for _ in range(npred):
                mask_e *= _sample_block_mask(p_size, duration, height, width, max_context_duration)

            # Flatten to get token indices
            mask_e = mask_e.flatten()

            # Get prediction and encoder token indices
            mask_p = torch.argwhere(mask_e == 0).squeeze()  # prediction tokens (masked)
            mask_e = torch.nonzero(mask_e).squeeze()        # encoder tokens (kept)

            # Ensure we have some context
            empty_context = len(mask_e) == 0
            if not empty_context:
                min_keep_pred = min(min_keep_pred, len(mask_p))
                min_keep_enc = min(min_keep_enc, len(mask_e))
                collated_masks_pred.append(mask_p)
                collated_masks_enc.append(mask_e)

    # Trim to minimum sizes to ensure consistent batch dimensions
    collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
    collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]

    # Convert to tensors and move to device
    collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc).to(device)
    collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred).to(device)

    return collated_masks_enc, collated_masks_pred

def compute_vjepa_loss(video_path, encoder, target_encoder, predictor,
                      img_size=256, context_frames=15, frames_per_clip=33, loss_exp=2):
    """
    Compute V-JEPA training-matched loss for a video.

    Args:
        video_path (str): Path to the input MP4 video
        encoder: Pre-loaded V-JEPA encoder model
        target_encoder: Pre-loaded V-JEPA target encoder model
        predictor: Pre-loaded V-JEPA predictor model
        img_size (int): Image size for processing
        context_frames (int): Number of initial frames to use as context
        frames_per_clip (int): Total frames in clip
        loss_exp (int): Exponent for loss calculation (default: 2 for L2 loss)

    Returns:
        float: V-JEPA training loss
    """
    set_deterministic()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process video
    video = get_video(video_path, max_frames=frames_per_clip)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).to(device)
    transform = build_pt_video_transform(img_size)
    x = transform(video_tensor).to(device).unsqueeze(0)  # [1, 3, 33, 256, 256]

    # Create clips and masks for training-matched pattern
    clips = x  # Single video tensor

    # Calculate mask positions
    grid_size = img_size // encoder.patch_size
    grid_depth = frames_per_clip // encoder.tubelet_size
    context_depth = context_frames // encoder.tubelet_size
    future_steps = grid_depth - context_depth

    N_context = int(grid_size**2 * context_depth)
    N_pred = int(grid_size**2 * future_steps)

    # Create position masks
    ctxt_positions = torch.arange(N_context, device=device).unsqueeze(0).repeat(1, 1)
    tgt_positions = torch.arange(N_pred, device=device).unsqueeze(0).repeat(1, 1)
    tgt_positions += N_context  # Offset by context size

    # Create masks exactly like training code
    masks_enc = ctxt_positions  # [B, N_context]
    masks_pred = tgt_positions  # [B, N_pred]

    # Training-matched forward functions
    def forward_target(c):
        h = target_encoder(c)
        h = torch.stack([F.layer_norm(hi, (hi.size(-1),)) for hi in h])
        return h

    def forward_context(c):
        z = encoder(c, masks_enc)
        z = predictor(z, masks_enc, masks_pred)
        return z

    def loss_fn(z, h):
        h = apply_masks(h, masks_pred, concat=False)

        loss, n = 0, 0
        for zi, hi in zip(z, h):
            for zij, hij in zip(zi, hi):
                loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                n += 1
        loss /= n
        return loss

    # Compute loss
    h = forward_target(clips)  # target features
    z = forward_context(clips)  # predictions
    loss = loss_fn(z, h)  # training-matched loss

    return loss

@torch.enable_grad()
def compute_vjepa_loss_from_tensor_unified(video_tensor, encoder, target_encoder, predictor,
                                          img_size=256, frames_per_clip=33, loss_exp=2,
                                          masking_mode="block", context_frames=15, mask_ratio=0.75,
                                          spatial_pred_mask_scale=(0.7, 0.7), temporal_pred_mask_scale=(1.0, 1.0),
                                          aspect_ratio=(0.75, 1.5), npred=2, max_context_frames_ratio=1.0,
                                          is_vae_output=True, seed=42):
    """
    Compute V-JEPA training-matched loss from a video tensor with configurable masking.

    Args:
        video_tensor (torch.Tensor): Video tensor of shape [B, C, T, H, W] or [C, T, H, W]
        encoder: Pre-loaded V-JEPA encoder model
        target_encoder: Pre-loaded V-JEPA target encoder model
        predictor: Pre-loaded V-JEPA predictor model
        img_size (int): Image size for processing
        frames_per_clip (int): Total frames in clip
        loss_exp (int): Exponent for loss calculation (default: 2 for L2 loss)
        masking_mode (str): "block" for V-JEPA block masking, "causal" for temporal masking, or "random" for random token masking
        context_frames (int): Number of context frames (only used for causal mode)
        mask_ratio (float): Ratio of tokens to mask (only used for random mode, 0.75 = mask 75%)
        spatial_pred_mask_scale (tuple): (min, max) spatial scale for prediction blocks (block mode only)
        temporal_pred_mask_scale (tuple): (min, max) temporal scale for prediction blocks (block mode only)
        aspect_ratio (tuple): (min, max) aspect ratio range for blocks (block mode only)
        npred (int): Number of prediction blocks to sample (block mode only)
        max_context_frames_ratio (float): Maximum fraction of frames that can be context (block mode only)
        is_vae_output (bool): If True, assumes input is VAE output in [-1, 1] range
        seed (int): Random seed for reproducible masking

    Returns:
        torch.Tensor: V-JEPA training loss
    """
    # set_deterministic()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = next(encoder.parameters()).dtype
    video_tensor = video_tensor.to(device=device, dtype=model_dtype)
    transform = build_pt_video_transform(img_size)

    # Handle VAE output conversion with proper batch support
    if is_vae_output:
        # Handle both single video and batch inputs
        if video_tensor.dim() == 4:  # [C, T, H, W] - single video
            video_tensor = video_tensor.unsqueeze(0)

        # Convert VAE output [-1,1] to [0,255] directly, preserving dtype
        video_255 = (video_tensor + 1.0) * 127.5  # [-1,1] → [0,255] directly
        batch_size = video_255.shape[0]
        # Process each video in the batch
        batch_processed = []

        for b in range(batch_size):
            video_tcthw = video_255[b].permute(1, 0, 2, 3).to(device)  # [T, C, H, W]
            video_normalized = transform(video_tcthw)
            batch_processed.append(video_normalized)
        x = torch.stack(batch_processed, dim=0).to(model_dtype)

    else:
        # Input is already in correct format (e.g., from dataset after transforms)
        x = video_tensor.to(device)

    # Create clips and masks for training-matched pattern
    clips = x  # [B, C, T, H, W]

    # Generate masks using the abstracted function
    ctxt_positions, tgt_positions = generate_vjepa_masks(
        masking_mode=masking_mode,
        batch_size=x.shape[0],
        img_size=img_size,
        frames_per_clip=frames_per_clip,
        encoder=encoder,
        context_frames=context_frames,
        mask_ratio=mask_ratio,
        device=device,
        spatial_pred_mask_scale=spatial_pred_mask_scale,
        temporal_pred_mask_scale=temporal_pred_mask_scale,
        aspect_ratio=aspect_ratio,
        npred=npred,
        max_context_frames_ratio=max_context_frames_ratio,
        seed=seed
    )

    # Create masks exactly like training code
    masks_enc = ctxt_positions  # [B, num_keep]
    masks_pred = tgt_positions  # [B, num_mask]

    # Training-matched forward functions
    def forward_target(c):

        h = target_encoder(c)
        h = torch.stack([F.layer_norm(hi, (hi.size(-1),)) for hi in h])
        return h

    def forward_context(c):
        with torch.no_grad():
            z = encoder(c, masks_enc)
            z = predictor(z, masks_enc, masks_pred)
            z = F.layer_norm(z, (z.size(-1),))
            return z

    def loss_fn(z, h):
        h = apply_masks(h, masks_pred, concat=False)
        loss, n = 0, 0
        for zi, hi in zip(z, h):
            for zij, hij in zip(zi, hi):
                loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                n += 1
        loss /= n
        return loss

    def loss_fn_v2(z, h):
        h = apply_masks(h, masks_pred, concat=False)
        loss = F.mse_loss(z, h[0], reduction="mean")
        return loss

    h = forward_target(clips)  # target features

    z = forward_context(clips)


    z = z.to(h.device)


    loss = loss_fn(z, h)

    return loss

# @torch.enable_grad()
def compute_vjepa_loss_sliding_window(video_tensor, encoder, target_encoder, predictor,
                                          img_size=256, window_size=16, loss_exp=2,
                                          masking_mode="causal", context_frames=8, mask_ratio=None,
                                          spatial_pred_mask_scale=None, temporal_pred_mask_scale=None,
                                          aspect_ratio=None, npred=None, max_context_frames_ratio=None,
                                          is_vae_output=True, seed=42, stride=2, mode='mean'):
    """
    Compute V-JEPA training-matched loss from a video tensor using sliding windows.
    Breaks 49-frame video into sub-chunks of 16 frames with sliding window approach.

    Args:
        video_tensor (torch.Tensor): Video tensor of shape [B, C, T, H, W] or [C, T, H, W]
        encoder: Pre-loaded V-JEPA encoder model
        target_encoder: Pre-loaded V-JEPA target encoder model
        predictor: Pre-loaded V-JEPA predictor model
        img_size (int): Image size for processing
        window_size (int): Frames per sliding window chunk (default: 16)
        loss_exp (int): Exponent for loss calculation (default: 2 for L2 loss)
        masking_mode (str): "block" for V-JEPA block masking, "causal" for temporal masking, or "random" for random token masking
        context_frames (int): Number of context frames (only used for causal mode)
        mask_ratio (float): Ratio of tokens to mask (only used for random mode, 0.75 = mask 75%)
        spatial_pred_mask_scale (tuple): (min, max) spatial scale for prediction blocks (block mode only)
        temporal_pred_mask_scale (tuple): (min, max) temporal scale for prediction blocks (block mode only)
        aspect_ratio (tuple): (min, max) aspect ratio range for blocks (block mode only)
        npred (int): Number of prediction blocks to sample (block mode only)
        max_context_frames_ratio (float): Maximum fraction of frames that can be context (block mode only)
        is_vae_output (bool): If True, assumes input is VAE output in [-1, 1] range
        seed (int): Random seed for reproducible masking
        stride (int): Stride for sliding window (default: 2)
        mode (str): How to aggregate losses from chunks - 'mean', 'max' (default: 'mean')

    Returns:
        torch.Tensor: V-JEPA training loss
    """
    # set_deterministic()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = next(encoder.parameters()).dtype
    video_tensor = video_tensor.to(device=device, dtype=model_dtype)
    transform = build_pt_video_transform(img_size)

    # Handle VAE output conversion with proper batch support
    if is_vae_output:
        # Handle both single video and batch inputs
        if video_tensor.dim() == 4:  # [C, T, H, W] - single video
            video_tensor = video_tensor.unsqueeze(0)

        # Convert VAE output [-1,1] to [0,255] directly, preserving dtype
        video_255 = (video_tensor + 1.0) * 127.5  # [-1,1] → [0,255] directly
        batch_size = video_255.shape[0]
        # Process each video in the batch
        batch_processed = []

        for b in range(batch_size):
            video_tcthw = video_255[b].permute(1, 0, 2, 3).to(device)  # [T, C, H, W]
            video_normalized = transform(video_tcthw)
            batch_processed.append(video_normalized)
        x = torch.stack(batch_processed, dim=0).to(model_dtype)

    else:
        # Input is already in correct format (e.g., from dataset after transforms)
        video_255 = video_tensor.to(device)
        batch_size = video_255.shape[0]
        # Process each video in the batch
        batch_processed = []

        for b in range(batch_size):
            video_tcthw = video_255[b].permute(1, 0, 2, 3).to(device)  # [T, C, H, W]
            video_normalized = transform(video_tcthw)
            batch_processed.append(video_normalized)
        x = torch.stack(batch_processed, dim=0).to(model_dtype)

    # Create sliding window chunks
    clips = x  # [B, C, T, H, W]

    # Create sliding windows exactly as in calculate_torch_vjepa_loss
    pieces = clips.unfold(2, window_size, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
    pieces = pieces.flatten(0, 1)
    pieces = rearrange(pieces, "b t c h w -> b c t h w")
    # print(f"pieces: {pieces.shape}")

    # Process chunks one by one for memory efficiency
    CHUNK_SIZE = 1
    chunk_losses = []

    for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
        chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]

        # Generate masks for this chunk
        ctxt_positions, tgt_positions = generate_vjepa_masks(
            masking_mode=masking_mode,
            batch_size=chunk.shape[0],
            img_size=img_size,
            frames_per_clip=window_size,
            encoder=encoder,
            context_frames=context_frames,
            mask_ratio=mask_ratio,
            device=device,
            spatial_pred_mask_scale=spatial_pred_mask_scale,
            temporal_pred_mask_scale=temporal_pred_mask_scale,
            aspect_ratio=aspect_ratio,
            npred=npred,
            max_context_frames_ratio=max_context_frames_ratio,
            seed=seed + chunk_id  # Vary seed per chunk for diversity
        )

        # Create masks for this chunk
        masks_enc = ctxt_positions  # [chunk_size, num_keep]
        masks_pred = tgt_positions  # [chunk_size, num_mask]

        # print(f"masks_enc shape: {masks_enc.shape}")
        # print(f"masks_pred shape: {masks_pred.shape}")

        # Training-matched forward functions for this chunk
        def forward_target(c):
            h = target_encoder(c)
            h = torch.stack([F.layer_norm(hi, (hi.size(-1),)) for hi in h])
            return h

        def forward_context(c):
            with torch.no_grad():
                z = encoder(c, masks_enc)
                z = predictor(z, masks_enc, masks_pred)
                z = F.layer_norm(z, (z.size(-1),))
                return z

        def loss_fn(z, h):
            h = apply_masks(h, masks_pred, concat=False)
            loss = 1 - F.cosine_similarity(z, h[0], dim=-1).mean()
            return loss

        # Compute features and loss for this chunk
        h_chunk = forward_target(chunk)  # target features
        z_chunk = forward_context(chunk)
        z_chunk = z_chunk.to(h_chunk.device)

        chunk_loss = loss_fn(z_chunk, h_chunk)
        chunk_losses.append(chunk_loss)

    # Aggregate losses from all chunks
    if mode == 'mean':
        loss = torch.mean(torch.stack(chunk_losses))
    elif mode == 'max':
        loss = torch.max(torch.stack(chunk_losses))
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mean' or 'max'")

    print(f"video_tensor shape: {video_tensor.shape}")
    print(f"number of chunks: {len(chunk_losses)}")
    print(f"aggregated loss ({mode}): {loss.item()} similarity: {1 - loss.item():.6f}")

    return loss
