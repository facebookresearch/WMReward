import torch
# from torchcodec.decoders import VideoDecoder
import torchvision.transforms as transforms
from transformers import AutoVideoProcessor, AutoModel
import numpy as np
import torch.nn.functional as F
from diffusers.utils import export_to_video, load_video
import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import copy

from decord import VideoReader

import sys
sys.path.append("/home/yjianhao/project/quentinecode/vjepa2")
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

def get_video(path):
    vr = VideoReader(path)
    num_frames = len(vr)
    frame_count = min(32, num_frames)
    frame_idx = np.arange(0, frame_count)
    video = vr.get_batch(frame_idx).asnumpy()
    return video

def get_time_masks(n_timesteps=16, spatial_size=(16, 16), temporal_size=2, spatial_dim=(224, 224), temporal_dim=16, as_bool=False):
    assert n_timesteps % temporal_size == 0
    x, y = spatial_dim
    t = temporal_dim
    
    num_patches_spatial = (x / spatial_size[0]) * (y / spatial_size[1])
    num_patches_time = t / temporal_size
    patches_n_timesteps = int(num_patches_spatial * n_timesteps // temporal_size)
    
    patch_idcs = torch.arange(start=0, end=int(num_patches_spatial * num_patches_time), dtype=int)
    if as_bool:
        mask_enc = patch_idcs < patches_n_timesteps
        mask_pred = patch_idcs >= patches_n_timesteps
        full_mask = patch_idcs >= 0
    else:
        mask_enc = patch_idcs[:patches_n_timesteps]
        mask_pred = patch_idcs[patches_n_timesteps:]
        full_mask = patch_idcs
    
    return mask_enc, mask_pred, full_mask

def get_sequential_mask(spatial_size, temporal_size, spatial_dim, temporal_dim, start_frame, kernel_size, context_window_size, as_bool=False):
    """
    Generates a sequential mask for the given sequence.

    Args:
        spatial_size (tuple): Spatial size of each patch.
        temporal_size (int): Temporal size of each patch.
        spatial_dim (tuple): Spatial dimensions of the input data.
        temporal_dim (int): Temporal dimension of the input data.
        start_frame (int): Starting frame of the context window.
        kernel_size (int): Size of the context window (in frames).
        context_window_size (int): Number of frames to consider as context within the window.
        as_bool (bool, optional): Whether to return masks as boolean tensors. Defaults to False.

    Returns:
        mask_enc: Mask for the encoder (context).
        mask_pred: Mask for the predictor (target).
        full_mask: Full mask for the entire sequence.
    """
    x, y = spatial_dim
    t = temporal_dim
    
    num_patches_spatial = (x / spatial_size[0]) * (y / spatial_size[1])
    num_patches_time = t / temporal_size
    
    # Calculate the total number of tokens
    total_tokens = int(num_patches_spatial * num_patches_time)
    
    # Calculate the number of tokens in the kernel
    kernel_tokens = int(kernel_size / temporal_size * num_patches_spatial)
    
    # Calculate the number of tokens in the encoder mask
    context_window_tokens = int(context_window_size / temporal_size * num_patches_spatial)
    
    # Calculate the starting token index of the context window
    start_token_idx = int(start_frame / temporal_size * num_patches_spatial)

    patch_idcs = torch.arange(start=0, end=total_tokens, dtype=int)
    
    if as_bool:
        pass
    else:
        # Initialize masks as index tensors
        mask_enc = torch.arange(start=start_token_idx, end=start_token_idx+context_window_tokens, dtype=int)
        mask_pred = torch.arange(start=start_token_idx+context_window_tokens, end=start_token_idx+kernel_tokens, dtype=int)
        full_mask = torch.arange(start=0, end=total_tokens, dtype=int)
    
    return mask_enc, mask_pred, full_mask


def apply_masks(x, masks, concat=True):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)

def get_sliding_window_score_based(video, model, processor, kernel_size, context_window_size, stride=2, return_form='arr', mode='max', require_grad=False):
    video = processor(video, return_tensors="pt").to(model.device)
    model.eval()
    B, T, C, H, W = video['pixel_values_videos'].shape
    device = model.device
    patch_size = 16
    is_mae=False
    spatial_dim = (H,W) 
    start_index_arr = np.arange(0, T - kernel_size + 1, stride)

    # predictor_model = model.deep_copy()
    target_encoder = copy.deepcopy(model)

    loss_arr = []
    for start_index in start_index_arr:
        video_slice = video['pixel_values_videos'][:,start_index:start_index+kernel_size]
        # print("slice shape",video_slice.shape)
        
        # Generate masks using proper sliding window approach (same as v2)
        # For each 16-frame window: first context_window_size FRAMES = context, rest = targets
        temporal_patches_in_window = kernel_size // 2  # tubelet_size = 2
        spatial_patches_per_frame = (spatial_dim[0] // patch_size) * (spatial_dim[1] // patch_size)
        
        # Convert context_window_size (frames) to temporal patches
        context_temporal_patches = context_window_size // 2  # tubelet_size = 2
        if context_window_size % 2 != 0:
            print(f"Warning: context_window_size ({context_window_size}) not divisible by tubelet_size (2)")
            context_temporal_patches = (context_window_size + 1) // 2  # Round up
        
        # Context: first context_temporal_patches temporal patches
        context_tokens = context_temporal_patches * spatial_patches_per_frame
        context_indices = torch.arange(0, context_tokens, dtype=torch.long)
        
        # Target: remaining temporal patches in the window
        target_start = context_tokens
        target_end = temporal_patches_in_window * spatial_patches_per_frame
        target_indices = torch.arange(target_start, target_end, dtype=torch.long)
        
        m = context_indices
        m_ = target_indices
        full_m = torch.arange(0, target_end, dtype=torch.long)

        full_m = full_m.unsqueeze(0).to(device)
        m = m.unsqueeze(0).to(device)
        m_ = m_.unsqueeze(0).to(device)

        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]
        full_mask = [full_m.repeat(B, 1)]

        h = model(pixel_values_videos=video_slice, skip_predictor=True).last_hidden_state
        normalize_targets = False
        if normalize_targets:
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
        targets = apply_masks(h, masks_pred, concat=False)

        # outputs = model(pixel_values_videos=video_slice, context_mask=masks_enc, target_mask=masks_pred)
        outputs = target_encoder(pixel_values_videos=video_slice, context_mask=masks_enc, target_mask=masks_pred)
        preds = outputs.predictor_output.last_hidden_state
        normalize_preds = False
        if normalize_preds:
            preds = F.layer_norm(preds, (preds.size(-1),))  # normalize over feature-dim  [B, N, D]

        preds = preds[0].view(B,-1,*preds[0].shape[1:]).unsqueeze(0)
        targets = targets[0].view(B,-1,*targets[0].shape[1:])
        
        if require_grad:
            loss = F.l1_loss(preds,targets,reduction="none").mean((1,2,3))
            loss_arr.append(loss)
        else:
            loss = F.l1_loss(preds,targets,reduction="none").mean((1,2,3)).detach()
            loss_arr.append(loss.item())

    if not loss_arr:
        raise RuntimeError("No valid losses computed - all windows failed")
    
    if require_grad:
        loss_tensor = torch.stack(loss_arr)
        if mode=='max':
            final_loss = torch.max(loss_tensor)
        elif mode == 'mean':
            final_loss = torch.mean(loss_tensor)

        return final_loss
    else:
        # Convert to numpy for non-gradient case
        loss_values = []
        for l in loss_arr:
            if isinstance(l, torch.Tensor):
                loss_values.append(l.cpu().item())  # Convert to scalar value
            else:
                loss_values.append(l)
        
        if mode == 'max':
            final_loss = np.max(loss_values)
        elif mode == 'mean':
            final_loss = np.mean(loss_values)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if return_form == 'arr':
            return final_loss, loss_values
        else:
            return final_loss

def get_sliding_window_score_based_v2(video, model, processor, kernel_size, context_window_size, stride=2, return_form='arr', mode='max', require_grad=False):
    """
    Improved version of get_sliding_window_score_based with proper model config usage and better masking.
    """
    
    # Handle processed tensors vs raw video
    if isinstance(video, torch.Tensor) and video.dim() == 4:
        print(f"Input video shape to get_sliding_window_score_based_v2: {video.shape}")
        # Check if this is [T, C, H, W] or [C, T, H, W] based on typical dimensions
        if video.shape[1] == 3:  # Likely [T, C, H, W] where C=3 for RGB
            # Video is in [T, C, H, W] format, just add batch dimension to get [B, T, C, H, W]
            video_processed = {'pixel_values_videos': video.unsqueeze(0)}
            print(f"Treating as [T, C, H, W] format")
        else:
            # Video is in [C, T, H, W] format, need to convert to [B, T, C, H, W]
            video_processed = {'pixel_values_videos': video.permute(1, 0, 2, 3).unsqueeze(0)}
            print(f"Treating as [C, T, H, W] format")
    else:
        # Use processor for raw video
        video_processed = processor(video, return_tensors="pt")
    
    video_processed = {k: v.to(model.device) for k, v in video_processed.items()}
    model.eval()
    
    B, T, C, H, W = video_processed['pixel_values_videos'].shape
    device = model.device
    
    # Use model configuration instead of hardcoded values
    try:
        patch_size = model.config.patch_size
        tubelet_size = model.config.tubelet_size if hasattr(model.config, 'tubelet_size') else 2
        # For processed tensors, use actual dimensions; for raw video use config
        spatial_dim = (H, W)
    except AttributeError:
        # Fallback to defaults if config doesn't have expected attributes
        print("Warning: Using fallback values for model config")
        patch_size = 16
        tubelet_size = 2
        spatial_dim = (H, W)
    
    print(f"Using patch_size={patch_size}, tubelet_size={tubelet_size}, spatial_dim={spatial_dim}")
    print(f"Video shape: {B, T, C, H, W}")
    
    # Validate inputs
    if context_window_size >= kernel_size:
        raise ValueError(f"context_window_size ({context_window_size}) must be less than kernel_size ({kernel_size})")
    
    if T < kernel_size:
        raise ValueError(f"Video length ({T}) must be >= kernel_size ({kernel_size})")
    
    # Calculate sliding window positions
    start_indices = np.arange(0, T - kernel_size + 1, stride)
    
    if len(start_indices) == 0:
        raise ValueError("No valid windows can be created with given parameters")
    
    loss_arr = []
    
    for start_index in start_indices:
        # Extract video slice
        video_slice = video_processed['pixel_values_videos'][:, start_index:start_index+kernel_size]
        
        # Generate masks using proper sliding window approach
        # For each 16-frame window: first context_window_size FRAMES = context, rest = targets
        temporal_patches_in_window = kernel_size // tubelet_size
        spatial_patches_per_frame = (spatial_dim[0] // patch_size) * (spatial_dim[1] // patch_size)
        
        # Convert context_window_size (frames) to temporal patches
        context_temporal_patches = context_window_size // tubelet_size
        if context_window_size % tubelet_size != 0:
            print(f"Warning: context_window_size ({context_window_size}) not divisible by tubelet_size ({tubelet_size})")
            context_temporal_patches = (context_window_size + tubelet_size - 1) // tubelet_size  # Round up
        
        # Context: first context_temporal_patches temporal patches
        context_tokens = context_temporal_patches * spatial_patches_per_frame
        context_indices = torch.arange(0, context_tokens, dtype=torch.long)
        
        # Target: remaining temporal patches in the window
        target_start = context_tokens
        target_end = temporal_patches_in_window * spatial_patches_per_frame
        target_indices = torch.arange(target_start, target_end, dtype=torch.long)
        
        context_frames = context_temporal_patches * tubelet_size
        target_frames = (temporal_patches_in_window - context_temporal_patches) * tubelet_size
        
        print(f"Window {start_index}: Context frames={context_frames}, Target frames={target_frames}")
        print(f"  Context patches={len(context_indices)}, Target patches={len(target_indices)}")
        print(f"  Context temporal patches: {context_temporal_patches}")
        print(f"  Target temporal patches: {temporal_patches_in_window - context_temporal_patches}")
        
        mask_enc = context_indices
        mask_pred = target_indices
        
        # Move masks to device and prepare for batching
        mask_enc = mask_enc.unsqueeze(0).to(device)
        mask_pred = mask_pred.unsqueeze(0).to(device)
        
        masks_enc = [mask_enc.repeat(B, 1)]
        masks_pred = [mask_pred.repeat(B, 1)]
        
        # Forward pass through encoder to get targets
        with torch.set_grad_enabled(require_grad):
            encoder_output = model(pixel_values_videos=video_slice, skip_predictor=True)
            h = encoder_output.last_hidden_state
            
            # Apply consistent normalization strategy
            normalize_features = True  # Make this configurable if needed
            if normalize_features:
                h = F.layer_norm(h, (h.size(-1),))
            
            # Apply masks to get targets
            targets = apply_masks(h, masks_pred, concat=False)
            
            # Forward pass through predictor
            predictor_output = model(
                pixel_values_videos=video_slice, 
                context_mask=masks_enc, 
                target_mask=masks_pred
            )
            preds = predictor_output.predictor_output.last_hidden_state
            
            # Apply same normalization to predictions
            if normalize_features:
                preds = F.layer_norm(preds, (preds.size(-1),))
            
            # Reshape for loss computation - ensure shapes match properly
            # preds comes as [B, seq_len, hidden_dim] from predictor output
            # targets comes as [B, seq_len, hidden_dim] from apply_masks
            preds_reshaped = preds[0]  # Shape: [seq_len, hidden_dim]
            targets_reshaped = targets[0]  # Shape: [B, seq_len, hidden_dim] -> need to squeeze batch dim
            
            if len(targets_reshaped.shape) == 3 and targets_reshaped.shape[0] == 1:
                targets_reshaped = targets_reshaped.squeeze(0)  # [seq_len, hidden_dim]
            
            # Compute loss - should return a scalar per window
            loss = F.l1_loss(preds_reshaped, targets_reshaped, reduction="mean")  # This gives us a single scalar
            
            if require_grad:
                loss_arr.append(loss)  # Keep as tensor for gradients
            else:
                loss_arr.append(loss.item())  # Convert to Python scalar
    
    if not loss_arr:
        raise RuntimeError("No valid losses computed - all windows failed")
    
    # Aggregate results
    if require_grad:
        loss_tensor = torch.stack(loss_arr)
        if mode == 'max':
            final_loss = torch.max(loss_tensor)
        elif mode == 'mean':
            final_loss = torch.mean(loss_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if return_form == 'arr':
            return final_loss, loss_tensor
        else:
            return final_loss
    else:
        # loss_arr already contains Python scalars
        if mode == 'max':
            final_loss = max(loss_arr)
        elif mode == 'mean':
            final_loss = sum(loss_arr) / len(loss_arr)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if return_form == 'arr':
            return final_loss, loss_arr
        else:
            return final_loss





IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

@torch.enable_grad()
def calculate_torch_vjepa_loss(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=False, mode='max', return_arr=False, is_vae_output=True, save_step=None):
    """
    Calculate V-JEPA loss exactly as in reproduce_intphys_clean.py.
    Uses sliding windows with proper batching and chunking.
    
    Args:
        video_tensor: Tensor of shape [B, C, T, H, W] where T > frames_per_clip
        model: Quentin's V-JEPA model
        context_length: Context length to use 
        frames_per_clip: Number of frames per clip (16 for V-JEPA)
        stride: Stride for sliding window
        use_bfloat16: Whether to use bfloat16 precision
        require_grad: Whether to keep gradients (True) or return detached value (False)
        mode: 'max' or 'mean' for aggregating losses across windows
        return_arr: Whether to return individual window losses
        is_vae_output: If True, converts from VAE output [-1,1] to [0,255]. If False, assumes input is already in correct format.
        save_step: If not None, saves loss_grad as './temp/invest/loss_grad_step{save_step}.npy'
    
    Returns:
        torch.Tensor or float: Max loss across all windows
    """
    from einops import rearrange
    import numpy as np
    import os
    
    model.eval()
    num_videos, C, T, H, W = video_tensor.shape
    device = next(model.parameters()).device

    # Conditional gradient context
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()
    
    with grad_context:
        
        if is_vae_output:
            transform = build_pt_video_transform(256)
    
            # Convert VAE output [-1,1] to [0,255] directly
            video_255 = (video_tensor + 1.0) * 127.5  # [-1,1] → [0,255] directly
            
            # Transform expects [T, C, H, W] format, so reshape from [B, C, T, H, W]
            B, C, T, H, W = video_255.shape
            video_tcthw = video_255.squeeze(0).permute(1, 0, 2, 3).to(device)  # [B, C, T, H, W] -> [T, C, H, W]
            video_normalized = transform(video_tcthw)
            # Transform outputs [T, C, H, W], convert back to [B, C, T, H, W] format
            video_normalized = video_normalized.unsqueeze(0)  # Add batch dim: [1, T, C, H, W]
            
            
        else:
            # Input is already in correct format (e.g., from IntPhys dataset after transforms)
            video_normalized = video_tensor.to(device)
        
        # Update model parameters exactly as in reproduce_intphys_clean.py
        model.nb_context_frames = context_length
        model.frames_per_clip = frames_per_clip
        model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
        
        # Create sliding windows exactly as in reproduce_intphys_clean.py  
        pieces = video_normalized.unfold(2, model.frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
        pieces = pieces.flatten(0, 1)
        pieces = rearrange(pieces, "b t c h w -> b c t h w")

        # Collect all predictions and targets exactly like reference implementation
        chunked_preds = []
        chunked_targets = []
        CHUNK_SIZE = 1  # Process one at a time for memory efficiency
        
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
        for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
            chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]
            
            preds, targets = model(chunk)
            chunked_preds.append(preds)
            chunked_targets.append(targets)
    
        # Combine all chunks exactly as in reference implementation
        preds = torch.vstack(chunked_preds)
        targets = torch.vstack(chunked_targets)
        preds = preds.view(num_videos, -1, *preds.shape[1:])
        targets = targets.view(num_videos, -1, *targets.shape[1:])
        
        # Compute loss exactly as Quentin does
        loss = F.l1_loss(preds, targets, reduction="none").mean((2, 3))
        
        # Recompute with gradients if needed
        preds_grad = torch.vstack([p.to(device) for p in chunked_preds])
        targets_grad = torch.vstack([t.to(device) for t in chunked_targets])
        preds_grad = preds_grad.view(num_videos, -1, *preds_grad.shape[1:])
        targets_grad = targets_grad.view(num_videos, -1, *targets_grad.shape[1:])
        preds_grad = preds_grad.detach()
        loss_grad = F.l1_loss(preds_grad, targets_grad, reduction="none").mean((2, 3))
        
        if mode == 'max':
            final_loss, max_idx = torch.max(loss_grad.view(-1), dim=0)
        elif mode == 'mean':
            final_loss = torch.mean(loss_grad)
        

        # Save loss_grad if requested
        # if save_step is not None:
        #     os.makedirs('./temp/invest', exist_ok=True)
        #     np.save(f'./temp/invest/loss_grad_step{save_step}.npy',)
        
        if require_grad:
            if return_arr:
                return final_loss, loss_grad.detach().cpu().numpy()
            else:
                return final_loss,  loss_grad.detach().cpu().numpy()
        else:
            if return_arr:
                return final_loss.detach().item(), loss_grad
            else:
                return final_loss.detach().item()  # Return float

def test_score(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=False, mode='max', return_arr=False, is_vae_output=True):
    """
    Test score function that computes MSE loss between input video tensor and all zeros.
    Has the same signature as calculate_torch_vjepa_loss but performs a simple MSE computation.
    
    Args:
        video_tensor: Tensor of shape [B, C, T, H, W] 
        model: Model (unused in this function, kept for signature compatibility)
        context_length: Context length (unused, kept for signature compatibility)
        frames_per_clip: Number of frames per clip (unused, kept for signature compatibility)
        stride: Stride (unused, kept for signature compatibility)
        use_bfloat16: Whether to use bfloat16 precision (unused, kept for signature compatibility)
        require_grad: Whether to keep gradients (True) or return detached value (False)
        mode: Aggregation mode (unused, kept for signature compatibility)
        return_arr: Whether to return array (unused, kept for signature compatibility)
        is_vae_output: Whether input is VAE output (unused, kept for signature compatibility)
    
    Returns:
        torch.Tensor or float: MSE loss between video_tensor and zeros
    """
    
    # Create zero tensor with same shape and device as input
    video_tensor = (video_tensor + 1.0) * 127.5
    zero_tensor = torch.zeros_like(video_tensor)
    
    # Conditional gradient context
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()
    
    with grad_context:
        # Compute MSE loss
        mse_loss = F.mse_loss(video_tensor, zero_tensor, reduction='mean')

        print(f"video_tensor: {video_tensor.norm(2).item()}")
        print(f"zero_tensor: {zero_tensor.norm(2).item()}")
        print(f"mse_loss: {mse_loss}")
        
        if require_grad:
            return mse_loss
        else:
            return mse_loss.detach().item()


def push_target_tensor(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=False, mode='max', return_arr=False, is_vae_output=True):
    """
    Push the video to a predefined V-JEPA tensor.
    """
    model.eval()
    device = next(model.parameters()).device
    num_videos, C, T, H, W = video_tensor.shape

    target_feature = torch.load("/home/yjianhao/project/video_guidance/vjepa_feature.pt")

    transform = build_pt_video_transform(256)

    # Convert VAE output [-1,1] to [0,255] directly
    video_255 = (video_tensor + 1.0) * 127.5  # [-1,1] → [0,255] directly
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()
    
    with grad_context:
        # Transform expects [T, C, H, W] format, so reshape from [B, C, T, H, W]
        B, C, T, H, W = video_255.shape
        video_normalized = video_255.squeeze(0).permute(1, 0, 2, 3).to(device)  # [B, C, T, H, W] -> [T, C, H, W]
        video_normalized = transform(video_normalized).unsqueeze(0)
        # Transform outputs [T, C, H, W], convert back to [B, C, T, H, W] format
        video_normalized = video_normalized[:, :, :-1, :, :].to(device)

        print(f"video_normalized shape: {video_normalized.shape}")
        print(f"video_normalized min: {video_normalized.min().item()}, max: {video_normalized.max().item()}")
        print(f"video_tensor (input) min: {video_tensor.min().item()}, max: {video_tensor.max().item()}")
        print(model)
        # import pdb; pdb.set_trace()



        video_feature = model(video_normalized)  # shape: [1, ...] (feature vector)
        target_feature = target_feature.to(video_feature.device)
        # Compute loss
        print(f"video_feature shape: {video_feature.shape}")
        print(f"target_feature shape: {target_feature.shape}")
        print(f"video_feature: {video_feature.min().item()}, {video_feature.max().item()}")
        print(f"target_feature: {target_feature.min().item()}, {target_feature.max().item()}")
        print(f"video_feature norm: {video_feature.norm(2).item()}")
        print(f"target_feature norm: {target_feature.norm(2).item()}")
        loss = F.mse_loss(video_feature, target_feature, reduction="mean")
    return loss


def push_target_pixel(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=False, mode='max', return_arr=False, is_vae_output=True):
    """
    Push the video to a predefined V-JEPA tensor.
    """
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()
    with grad_context:
        num_videos, C, T, H, W = video_tensor.shape

        target_video_path = "/home/yjianhao/project/video_guidance/generated_videos/subject_consistency/vanilla_f33_s50_cfg5.0/a bicycle accelerating to gain speed.mp4"
        target_video = get_video(target_video_path)
        target_video_tensor = torch.from_numpy(target_video).permute(0, 3, 1, 2).unsqueeze(0).to(video_tensor.dtype)  # T x C x H x W, convert to float
        target_video_tensor = target_video_tensor.to(video_tensor.device)

        # Convert VAE output [-1,1] to [0,255] directly
        video_255 = (video_tensor + 1.0) * 127.5  # [-1,1] → [0,255] directly
        video_tensor = video_255[:, :, :-1, :, :].permute(0, 2, 1, 3, 4)

        # print(f"video_tensor shape: {video_tensor.shape}")
        # print(f"target_video_tensor shape: {target_video_tensor.shape}")
        # print(f"video_tensor min: {video_tensor.min().item()}, max: {video_tensor.max().item()}")
        # print(f"target_video_tensor min: {target_video_tensor.min().item()}, max: {target_video_tensor.max().item()}")
        # print(f"video_tensor norm: {video_tensor.norm(2).item()}")
        # print(f"target_video_tensor norm: {target_video_tensor.norm(2).item()}")
        loss = F.mse_loss(video_tensor, target_video_tensor, reduction="mean")

    return loss
    
