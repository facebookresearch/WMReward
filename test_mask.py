import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
import numpy as np
import torch.nn.functional as F

def get_time_masks(n_timesteps=4, spatial_size=(16, 16), temporal_size=2, spatial_dim=(224, 224), temporal_dim=16, as_bool=False):
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



import torch

def get_sequential_mask(spatial_size, temporal_size, spatial_dim, temporal_dim, start_frame, context_window, k, as_bool=False):
    """
    Generates a sequential mask for the given sequence.

    Args:
        spatial_size (tuple): Spatial size of each patch.
        temporal_size (int): Temporal size of each patch.
        spatial_dim (tuple): Spatial dimensions of the input data.
        temporal_dim (int): Temporal dimension of the input data.
        start_frame (int): Starting frame of the context window.
        context_window (int): Size of the context window (in frames).
        k (int): Number of frames to consider as context within the window.
        as_bool (bool, optional): Whether to return masks as boolean tensors. Defaults to False.

    Returns:
        mask_enc: Mask for the encoder (context).
        mask_pred: Mask for the predictor (target).
        full_mask: Full mask for the entire sequence.
    """
    x, y = spatial_dim
    t = temporal_dim
    # print(x,y,t)
    
    num_patches_spatial = (x / spatial_size[0]) * (y / spatial_size[1])
    num_patches_time = t / temporal_size
    # print(num_patches_spatial, num_patches_time)
    
    # Calculate the total number of tokens
    total_tokens = int(num_patches_spatial * num_patches_time)
    
    # Calculate the number of tokens in the context window
    context_window_tokens = int(context_window / temporal_size * num_patches_spatial)
    # print(total_tokens, context_window_tokens)
    
    # Calculate the number of tokens in the encoder mask
    k_tokens = int(k / temporal_size * num_patches_spatial)
    
    # Calculate the starting token index of the context window
    start_token_idx = int(start_frame / temporal_size * num_patches_spatial)
    # print(start_token_idx, k_tokens)

    patch_idcs = torch.arange(start=0, end=total_tokens, dtype=int)
    
    if as_bool:
        pass
    else:
        # Initialize masks as index tensors
        mask_enc = torch.arange(start=start_token_idx, end=start_token_idx+k_tokens, dtype=int)
        mask_pred = torch.arange(start=start_token_idx+k_tokens, end=start_token_idx+context_window_tokens, dtype=int)
        full_mask = torch.arange(start=0, end=total_tokens, dtype=int)
    
    return mask_enc, mask_pred, full_mask

nframes = 16
# mask_enc, mask_pred, full_mask = get_time_masks(n_timesteps=2, spatial_size=(16, 16), temporal_size=2, spatial_dim=(224, 224), temporal_dim=nframes, as_bool=False)
# print("mask_enc:", mask_enc)
# print("mask_pred:", mask_pred)
# print("full_mask:", full_mask)

mask_enc, mask_pred, full_mask = get_sequential_mask(spatial_size=(16, 16), temporal_size=2, spatial_dim=(224, 224), temporal_dim=nframes, start_frame=0, context_window=4, k=2, as_bool=False)
print("mask_enc:", mask_enc, mask_enc.shape)
print("mask_pred:", mask_pred, mask_pred.shape)
print("full_mask:", full_mask.shape)