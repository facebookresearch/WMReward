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

# Add V-JEPA path
sys.path.append("./vjepa2")
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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def _clean_backbone_key(state_dict):
    for key, val in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        state_dict[key] = val
    return state_dict

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
    img_size = 384 if "384" in model else 256
    if model == "vith" or model == "vit_huge":
        encoder = vit_huge_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = "/home/yjianhao/project/checkpoints/vith.pt"

    elif model == "vitg" or model == "vit_giant":
        encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = "/home/yjianhao/project/checkpoints/vitg.pt"
        # model_path = "/home/yjianhao/project/checkpoints/vjepa2-ac-vitg.pt"
    elif model == "vitg384" or model == "vit_giant_384":
        encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = "/home/yjianhao/.cache/torch/hub/checkpoints/vitg-384.pt"
    elif model == "vitgac" or model == "vit_giant_ac":
        encoder = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=num_frames)
        model_path = "/home/yjianhao/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt"
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
        print(f"h: {h[0].shape}")
        print(f"z: {z.shape}")
        loss = F.mse_loss(z, h[0], reduction="mean")
        return loss
    
    h = forward_target(clips)  # target features 
    
    z = forward_context(clips)

    
    z = z.to(h.device)

    print(f"video_tensor shape: {video_tensor.shape}")
    print(f"video_feature shape: {z.shape}")
    print(f"target_feature shape: {h.shape}")
    print(f"video_feature: {z.min().item()}, {z.max().item()}")
    print(f"target_feature: {h.min().item()}, {h.max().item()}")
    print(f"video_feature norm: {z.norm(2).item()}")
    print(f"target_feature norm: {h.norm(2).item()}")


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

        # def loss_fn(z, h):
        #     h = apply_masks(h, masks_pred, concat=False)
        #     loss, n = 0, 0
        #     for zi, hi in zip(z, h):
        #         for zij, hij in zip(zi, hi):
        #             loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
        #             n += 1
        #     loss /= n
        #     return loss

        def loss_fn_v2(z, h):
            h = apply_masks(h, masks_pred, concat=False)
            loss = 1 - F.cosine_similarity(z, h[0], dim=1).mean()
            return loss
        
        # Compute features and loss for this chunk
        h_chunk = forward_target(chunk)  # target features 
        z_chunk = forward_context(chunk)
        z_chunk = z_chunk.to(h_chunk.device)
        
        chunk_loss = loss_fn_v2(z_chunk, h_chunk)
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

def push_target_tensor_v2(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=True, mode='max', return_arr=False, is_vae_output=True):
    """
    Push the video to a V-JEPA feature computed from a target video loaded each time.
    """
    model.eval()
    device = next(model.parameters()).device
    # Load target video and compute its feature
    # target_video_path = "/home/yjianhao/project/video_guidance/0160_full-videos_30FPS_perspective-left_take-1_trimmed-single-cradle.mp4"
    target_video_path = "/home/yjianhao/project/video_guidance/cradle_49.mp4"
    # target_video_path = "/home/yjianhao/project/video_guidance/red_49.mp4"
    # target_video_path = "/home/yjianhao/project/video_guidance/results/action/rho_scale_0.1/gt_video.mp4"
    # target_video_path = "/home/yjianhao/project/video_guidance/results/action/rho_scale_1/gt_video.mp4"
    # target_video_path = "/home/yjianhao/project/video_guidance/sampled1.mp4"
    # target_video_path = "/home/yjianhao/project/video_guidance/sampled.mp4"
    num_video_frames = video_tensor.shape[2]  # T dimension from video_tensor [B, C, T, H, W]
    
    # Load target video normally
    target_video = get_video(target_video_path, max_frames=49)
    target_video_tensor = torch.from_numpy(target_video).permute(0, 3, 1, 2).to(device).float()

    # print(f"target_video_tensor shape: {target_video_tensor.shape}")
    # print(f"video_tensor shape: {video_tensor.shape}")

    transform = build_pt_video_transform(256)

    # Apply post-processing based on input type
    if is_vae_output:
        # For VAE output: [-1,1] → [0,1] → [0,255]
        video_postprocessed = (video_tensor * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1] with clamp
        video_255 = video_postprocessed * 255.0  # [0,1] → [0,255]
    else:
        # Already post-processed: [0,255] (no further processing needed)
        video_255 = video_tensor
    
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()

    def forward_target(c):
        h = model(c)
        h = torch.stack([F.layer_norm(hi, (hi.size(-1),)) for hi in h])
        return h
    
    with grad_context:
        # Process input video feature
        # Transform expects [T, C, H, W] format, so reshape from [B, C, T, H, W]
        B, C, T, H, W = video_255.shape
        video_normalized = video_255.squeeze(0).permute(1, 0, 2, 3).to(device)  # [B, C, T, H, W] -> [T, C, H, W]
        video_normalized = transform(video_normalized).unsqueeze(0).to(device)

        # Process target video feature
        target_normalized = transform(target_video_tensor).to(device).unsqueeze(0)
        
        # DEBUG: Show post-transform ranges
        print(f"video_normalized shape: {video_normalized.shape}")
        print(f"target_normalized shape: {target_normalized.shape}")
        print(f"  After guidance transform range: [{video_normalized.min().item():.3f}, {video_normalized.max().item():.3f}]")
        print(f"  Target after transform range: [{target_normalized.min().item():.3f}, {target_normalized.max().item():.3f}]")

        # Compute features
        video_feature = forward_target(video_normalized)
        target_feature = forward_target(target_normalized)
         # Ensure both features are on the same device
        target_feature = target_feature.to(video_feature.device)
        
        # Compute loss
        # print(f"video_feature shape: {video_feature.shape}")
        # print(f"target_feature shape: {target_feature.shape}")
        # loss = F.mse_loss(video_feature, target_feature, reduction="mean")
        # loss = F.l1_loss(video_feature, target_feature, reduction="mean")
        # print(f"video_feature shape: {video_feature.shape}")
        loss = 1 - F.cosine_similarity(video_feature, target_feature, dim=1).mean()
        print(f"sim shape: {F.cosine_similarity(video_feature, target_feature).shape}")
        print(f"similarity: {F.cosine_similarity(video_feature, target_feature, dim=1).mean().item():.6f}")

        # print(f"loss: {loss.item():.6f}")
        

    return loss


# def push_target_tensor_dinov2(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=True, mode='max', return_arr=False, is_vae_output=True):
#     """
#     Push the video to a DINOv2 feature computed from a target video loaded each time.
#     Processes videos frame-by-frame since DINOv2 is designed for images.
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     # target_video_path = "/home/yjianhao/project/video_guidance/sampled.mp4"
#     # target_video_path = "/home/yjianhao/project/video_guidance/cradle_49.mp4"
#     # target_video_path = "/home/yjianhao/project/video_guidance/sampled.mp4"
#     target_video_path = "/home/yjianhao/project/video_guidance/red_49.mp4"
#     # target_video_path = "/home/yjianhao/project/video_guidance/sampled2.mp4"
#     num_video_frames = video_tensor.shape[2]  # T dimension from video_tensor [B, C, T, H, W]
    
#     # Load target video normally
#     target_video = get_video(target_video_path, max_frames=49)
#     target_video_tensor = torch.from_numpy(target_video).permute(0, 3, 1, 2).to(device).float()

#     print(f"target_video_tensor shape: {target_video_tensor.shape}")
#     print(f"video_tensor shape: {video_tensor.shape}")
    
#     transform = build_dinov2_transform()

#     # Apply post-processing based on input type
#     if is_vae_output:
#         # For VAE output: [-1,1] → [0,1] → [0,255]
#         video_postprocessed = (video_tensor * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1] with clamp
#         video_255 = video_postprocessed * 255.0  # [0,1] → [0,255]
#     else:
#         # Already post-processed: [0,255] (no further processing needed)
#         video_255 = video_tensor
    
#     grad_context = torch.enable_grad() if require_grad else torch.no_grad()

#     def forward_video_dinov2(video_frames, model, transform):
#         """
#         Process video frames through DINOv2 frame-by-frame and concatenate features.
        
#         Args:
#             video_frames: [T, C, H, W] tensor of video frames (values in [0, 255])
#             model: DINOv2 model
#             transform: DINOv2 transform
            
#         Returns:
#             Concatenated video features [T * feature_dim]
#         """
#         frame_features = []
        
#         for t in range(video_frames.shape[0]):
#             # Extract single frame: [C, H, W]
#             frame = video_frames[t]
            
#             # print(f"frame shape: {frame.shape}")
            
#             # 2. Resize to DINOv2 expected size (224x224) - dimensions must be multiples of patch size 14
#             C, H, W = frame.shape
#             if H != 224 or W != 224:
#                 frame = F.interpolate(
#                     frame.unsqueeze(0),  # Add batch dim: [1, C, H, W]
#                     size=(224, 224),
#                     mode='bilinear',
#                     align_corners=False
#                 ).squeeze(0)  # Remove batch dim: [C, H, W]

#             # print(f"frame shape 2: {frame.shape}")
            
#             # 3. Convert from [0, 255] to [0, 1] (since ToTensor() expects [0, 255] -> [0, 1])
#             frame_normalized = frame / 255.0
            
#             # 4. Apply DINOv2 normalization: scale by 255 and normalize
#             # DINOv2 transform does: lambda x: 255.0 * x[:3] then normalize
#             frame_scaled = frame_normalized * 255.0
            
#             # 5. Apply DINOv2 normalization with its specific mean/std
#             mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
#             std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)
#             frame_transformed = (frame_scaled - mean) / std
            
#             # Add batch dimension for model: [1, C, H, W]
#             frame_batch = frame_transformed.unsqueeze(0)
            
#             # Get features from DINOv2
#             with torch.no_grad() if not frame_batch.requires_grad else torch.enable_grad():
#                 frame_feature = model(frame_batch)  # [1, feature_dim]
            
#             frame_features.append(frame_feature.squeeze(0))  # Remove batch dim: [feature_dim]
        
#         # Concatenate all frame features: [T * feature_dim]
#         concatenated_features = torch.cat(frame_features, dim=0)
            
#         return concatenated_features
    
#     with grad_context:
#         # Process input video feature
#         # Reshape from [B, C, T, H, W] to [T, C, H, W]
#         B, C, T, H, W = video_255.shape
#         video_frames = video_255.squeeze(0).permute(1, 0, 2, 3)  # [B, C, T, H, W] -> [T, C, H, W]
        
#         # Process target video feature  
#         target_frames = target_video_tensor  # Already [T, C, H, W]
        
#         # DEBUG: Show pre-transform ranges
#         print(f"  Video frames range before transform: [{video_frames.shape} {video_frames.min().item():.3f}, {video_frames.max().item():.3f}]")
#         print(f"  Target frames range before transform: [{target_frames.shape} {target_frames.min().item():.3f}, {target_frames.max().item():.3f}]")

#         # Get DINOv2 features for video and target
#         video_feature = forward_video_dinov2(video_frames, model, transform)
#         target_feature = forward_video_dinov2(target_frames, model, transform)
        
#         # Ensure both features are on the same device
#         target_feature = target_feature.to(video_feature.device)
        
#         # DEBUG: Show feature shapes and ranges
#         print(f"  Video feature shape: {video_feature.shape}, range: [{video_feature.min().item():.3f}, {video_feature.max().item():.3f}]")
#         print(f"  Target feature shape: {target_feature.shape}, range: [{target_feature.min().item():.3f}, {target_feature.max().item():.3f}]")

#         # Compute loss
#         loss = F.l1_loss(video_feature, target_feature, reduction="mean")
#         print(f"  DINOv2 guidance loss: {loss.item():.6f}")

#     return loss


@torch.enable_grad()
def push_target_tensor_dinov2(
    video_tensor: torch.Tensor,               # [1,3,T,H,W] (your case) or [B,3,T,H,W]
    model,                                     # DINOv2 backbone (e.g., dinov2_vitl14_reg)
    target_video_path: str = "/home/yjianhao/project/video_guidance/red_49.mp4",                    # e.g. "/home/.../red_49.mp4"
    is_vae_output: bool = True,                # True if video_tensor is in [-1,1]
    assume_target_bgr: bool = False,           # True if get_video() loads via OpenCV (BGR)
    use_derivative: bool = False               # add temporal-derivative cosine
) -> torch.Tensor:
    """
    Differentiable DINOv2 guidance loss between generated frames and target video.
    Everything inline: shape handling, preprocessing, CLS features, cosine loss.
    """
    device = next(model.parameters()).device
    dtype = torch.float32
    model.eval()

    # ---- current video -> [T,C,H,W] in 0..255 RGB ----
    x = video_tensor.to(device=device, dtype=dtype)
    if is_vae_output:
        x = (x * 0.5 + 0.5).clamp(0, 1) * 255.0           # [-1,1] -> [0,255]
    if x.ndim == 5:                                       # [B,C,T,H,W]
        if x.size(0) != 1:
            # pick first item; change to mean over batch if you want multi-B support
            x = x[0]                                      # [C,T,H,W]
        else:
            x = x[0]                                      # [C,T,H,W]
        vid_tchw = x.permute(1, 0, 2, 3).contiguous()     # [T,C,H,W]
    elif x.ndim == 4:                                     # [C,T,H,W]
        vid_tchw = x.permute(1, 0, 2, 3).contiguous()     # [T,C,H,W]
    else:
        raise ValueError(f"Expected 4D/5D, got {tuple(x.shape)}")

    T = vid_tchw.shape[0]

    # ---- target -> [T,C,H,W] in 0..255 RGB ----
    tgt_np = get_video(target_video_path, max_frames=T)   # [T,H,W,C] uint8
    tgt = torch.from_numpy(tgt_np).permute(0, 3, 1, 2).to(device=device, dtype=dtype)  # [T,C,H,W]
    if assume_target_bgr:
        tgt = tgt[:, [2, 1, 0], :, :]                     # BGR -> RGB

    # ---- preprocess for DINOv2 (0..255 domain mean/std), resize to 224 ----
    def preprocess_224_255rgb(z: torch.Tensor) -> torch.Tensor:
        z = F.interpolate(z, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([123.675, 116.28, 103.53], device=z.device, dtype=z.dtype).view(1, 3, 1, 1)
        std  = torch.tensor([58.395,  57.12,  57.375], device=z.device, dtype=z.dtype).view(1, 3, 1, 1)
        return (z - mean) / std

    vid_in = preprocess_224_255rgb(vid_tchw)              # [T,3,224,224]
    with torch.no_grad():
        tgt_in = preprocess_224_255rgb(tgt)               # [T,3,224,224]

    # ---- forward CLS features (batched over T) ----
    def cls_feats(m, inp: torch.Tensor) -> torch.Tensor:
        out = m.forward_features(inp)                     # dict for official DINOv2
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                f = out["x_norm_clstoken"]               # [T,D] (reg checkpoints)
            elif "x_norm_cls" in out:
                f = out["x_norm_cls"]
            else:
                # fallback: first tensor value
                for v in out.values():
                    if torch.is_tensor(v):
                        f = v
                        break
                else:
                    raise RuntimeError("forward_features returned no tensor outputs")
        else:
            f = out                                       # rare forks
        return F.normalize(f, dim=-1)                     # [T,D], L2-normalized

    f_vid = cls_feats(model, vid_in)                      # [T,D], requires_grad
    with torch.no_grad():
        f_tgt = cls_feats(model, tgt_in)                  # [T,D]

    # ---- cosine loss (+ optional temporal derivative) ----
    print(f"f_vid shape: {f_vid.shape}")
    print(f"f_tgt shape: {f_tgt.shape}")
    feat_loss = 1.0 - F.cosine_similarity(f_vid, f_tgt, dim=-1).mean()

    return feat_loss




@torch.enable_grad()
def lpip_loss(video_tensor, lpips):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure LPIPS model is on the correct device
    lpips = lpips.to(device)
    
    # Normalization parameters for ImageNet - shaped for individual frames [1, 3, 1, 1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    target_video_path = "/home/yjianhao/project/video_guidance/sampled.mp4"
    target_video = get_video(target_video_path, max_frames=5)
    target_video_tensor = torch.from_numpy(target_video).permute(0, 3, 1, 2).to(device).float()

    # Preprocess video tensor: [-1,1] → [0,1]
    video_postprocessed = (video_tensor * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1] with clamp
    target_postprocessed = (target_video_tensor / 255.0).clamp(0, 1)  # [0,255] → [0,1]
    
    # Get dimensions
    B, C, T, H, W = video_postprocessed.shape
    T_target, C_target, H_target, W_target = target_postprocessed.shape
    
    # Ensure we have the same number of frames
    num_frames = min(T, T_target)
    
    total_loss = 0.0
    
    # Process each frame individually
    for t in range(num_frames):
        # Extract single frames [B, C, H, W]
        video_frame = video_postprocessed[:, :, t, :, :]  # [B, C, H, W]
        target_frame = target_postprocessed[t:t+1, :, :, :].expand(B, -1, -1, -1)  # [B, C, H, W]
        
        # Resize frames to 256x256
        video_frame_resized = F.interpolate(video_frame, size=(256, 256), mode='bilinear', align_corners=False)
        target_frame_resized = F.interpolate(target_frame, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Normalize for ImageNet
        video_frame_normalized = (video_frame_resized - mean) / std
        target_frame_normalized = (target_frame_resized - mean) / std
        
        # Compute LPIPS for this frame pair
        frame_loss = lpips(video_frame_normalized, target_frame_normalized)
        total_loss += frame_loss
    
    # Average loss across all frames
    avg_loss = total_loss / num_frames
    print(f"  LPIPS loss (avg across {num_frames} frames): {avg_loss.item():.6f}")

    return avg_loss

