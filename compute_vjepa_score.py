import torch
# from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
import numpy as np
import torch.nn.functional as F
from diffusers.utils import export_to_video, load_video
import os
import tempfile
from PIL import Image

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

def get_score(video,model,processor, context_length=2, frames_per_clip=33, require_grad=False):
    print("Video shape input", video.shape)
    video = processor(video, return_tensors="pt").to(model.device)
    model.eval()

    B, T, C, H, W = video['pixel_values_videos'].shape
    print(f"Video shape: {B, T, C, H, W}")
    device = model.device
    patch_size = 16
    temporal_size = 2
    is_mae=False
    frames_per_clip = T
    spatial_dim = (H, W) 
    num_videos = B
    m, m_, full_m = get_time_masks(n_timesteps=context_length, spatial_size=(patch_size, patch_size), temporal_dim=frames_per_clip, as_bool=is_mae)

    full_m = full_m.unsqueeze(0).to(device)
    m = m.unsqueeze(0).to(device)
    m_ = m_.unsqueeze(0).to(device)

    masks_enc = [m.repeat(B, 1)]
    masks_pred = [m_.repeat(B, 1)]
    full_mask = [full_m.repeat(B, 1)]

    h = model(**video, skip_predictor=True).last_hidden_state
    normalize_targets = True
    if normalize_targets:
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
    targets = apply_masks(h, masks_pred, concat=False)


    outputs = model(**video, context_mask=masks_enc, target_mask=masks_pred)
    preds = outputs.predictor_output.last_hidden_state



    preds = preds[0].view(num_videos, -1, *preds[0].shape[1:])
    targets = targets[0].view(num_videos, -1, *targets[0].shape[1:]).squeeze(0)
    loss = F.l1_loss(preds, targets, reduction="none").mean()

    if require_grad:
        return loss

    return loss.detach().item()

def get_sliding_window_score_based(video, model, processor, kernel_size, context_window_size, stride=2, return_form='arr', mode='max', require_grad=False):
    video = processor(video, return_tensors="pt").to(model.device)
    model.eval()
    B, T, C, H, W = video['pixel_values_videos'].shape
    device = model.device
    patch_size = 16
    is_mae=False
    spatial_dim = (H,W) 
    start_index_arr = np.arange(0, T - kernel_size + 1, stride)

    loss_arr = []
    for start_index in start_index_arr:
        video_slice = video['pixel_values_videos'][:,start_index:start_index+kernel_size]
        # print("slice shape",video_slice.shape)
        m, m_, full_m = get_time_masks(n_timesteps=context_window_size, spatial_size=(patch_size, patch_size), temporal_dim=kernel_size, as_bool=is_mae)

        full_m = full_m.unsqueeze(0).to(device)
        m = m.unsqueeze(0).to(device)
        m_ = m_.unsqueeze(0).to(device)

        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]
        full_mask = [full_m.repeat(B, 1)]

        h = model(pixel_values_videos=video_slice, skip_predictor=True).last_hidden_state
        normalize_targets = True
        if normalize_targets:
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
        targets = apply_masks(h, masks_pred, concat=False)

        outputs = model(pixel_values_videos=video_slice, context_mask=masks_enc, target_mask=masks_pred)
        preds = outputs.predictor_output.last_hidden_state

        preds = preds[0].view(B,-1,*preds[0].shape[1:]).unsqueeze(0)
        targets = targets[0].view(B,-1,*targets[0].shape[1:])
        
        if require_grad:
            loss = F.l1_loss(preds,targets,reduction="none").mean((1,2,3))
            loss_arr.append(loss)
        else:
            loss = F.l1_loss(preds,targets,reduction="none").mean((1,2,3)).detach()
            loss_arr.append(loss.item())

    if require_grad:
        loss_tensor = torch.stack(loss_arr)
        if mode=='max':
            final_loss = torch.max(loss_tensor)
        elif mode == 'mean':
            final_loss = torch.mean(loss_tensor)

        return final_loss
    else:
        # Original logic for non-gradient case
        if mode=='max':
            final_loss = np.max(loss_arr)
        elif mode == 'mean':
            final_loss = np.mean(loss_arr)

        if return_form == 'arr':
            return final_loss, loss_arr
        else:
            return final_loss

def get_sliding_window_score(video, model, processor, kernel_size, context_window_size, stride=2):
    video = processor(video, return_tensors="pt").to(model.device)
    model.eval()

    device = model.device
    patch_size = 16
    is_mae=False
    frames_per_clip = 33
    spatial_dim = (256, 256) 
    num_videos = 1
    B = num_videos
    start_index_arr = np.arange(0, frames_per_clip - kernel_size + 1, stride)

    loss_arr = []
    for start_index in start_index_arr:
        m, m_, full_m = get_sequential_mask(spatial_size=(patch_size, patch_size), temporal_size=2, spatial_dim=spatial_dim, temporal_dim=frames_per_clip, start_frame=start_index, kernel_size=kernel_size, context_window_size=context_window_size, as_bool=False)

        full_m = full_m.unsqueeze(0).to(device)
        m = m.unsqueeze(0).to(device)
        m_ = m_.unsqueeze(0).to(device)

        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]

        h = model(**video, skip_predictor=True).last_hidden_stateclear
        
        targets = apply_masks(h, masks_pred, concat=False)

        outputs = model(**video, context_mask=masks_enc, target_mask=masks_pred)
        preds = outputs.predictor_output.last_hidden_state

        preds = preds[0].view(num_videos, -1, *preds[0].shape[1:])
        targets = targets[0].view(num_videos, -1, *targets[0].shape[1:]).squeeze(0)
        loss = F.l1_loss(preds, targets, reduction="none").mean().detach()

        loss_arr.append(loss.item())
    
    loss_avr = np.mean(loss_arr)

    return loss_avr

def get_sliding_window_score_max_v2(video, model, processor, kernel_size, context_window_size, stride=2, loss_form="mean"):
    torch.set_grad_enabled(True)  # Enable gradient computation for the model
    video = processor(video, return_tensors="pt").to(model.device)
    model.eval()

    B, T, C, H, W = video['pixel_values_videos'].shape
    device = model.device
    patch_size = 16
    temporal_size = 2
    is_mae=False
    frames_per_clip = T
    spatial_dim = (H, W) 
    num_videos = B
    B = num_videos
    start_index_arr = np.arange(0, frames_per_clip - kernel_size + 1, stride)

    total_loss = None
    n_count = 0
    for start_index in start_index_arr:
        m, m_, full_m = get_sequential_mask(spatial_size=(patch_size, patch_size), temporal_size=2, spatial_dim=spatial_dim, temporal_dim=frames_per_clip, start_frame=start_index, kernel_size=kernel_size, context_window_size=context_window_size, as_bool=False)

        full_m = full_m.unsqueeze(0).to(device)
        m = m.unsqueeze(0).to(device)
        m_ = m_.unsqueeze(0).to(device)

        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]

        h = model(**video, skip_predictor=True).last_hidden_state
        targets = apply_masks(h, masks_pred, concat=False)

        outputs = model(**video, context_mask=masks_enc, target_mask=masks_pred)
        preds = outputs.predictor_output.last_hidden_state

        preds = preds[0].view(num_videos, -1, *preds[0].shape[1:])
        targets = targets[0].view(num_videos, -1, *targets[0].shape[1:]).squeeze(0)
        loss = F.l1_loss(preds, targets, reduction="none").mean()
        # print(f"Start index: {start_index}, Loss: {loss.item()}")  # Debugging line to inspect loss values
        if total_loss:
            total_loss += loss
        else:
            total_loss = loss

        n_count += 1

    final_loss = total_loss / n_count
    return final_loss

# def calculate_torch_vjepa_loss(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=False, mode='max', return_arr=False, is_vae_output=True):
#     """
#     Calculate V-JEPA loss exactly as in reproduce_intphys_clean.py.
#     Uses sliding windows with proper batching and chunking.
    
#     Args:
#         video_tensor: Tensor of shape [B, C, T, H, W] where T > frames_per_clip
#         model: Quentin's V-JEPA model
#         context_length: Context length to use 
#         frames_per_clip: Number of frames per clip (16 for V-JEPA)
#         stride: Stride for sliding window
#         use_bfloat16: Whether to use bfloat16 precision
#         require_grad: Whether to keep gradients (True) or return detached value (False)
#         mode: 'max' or 'mean' for aggregating losses across windows
#         return_arr: Whether to return individual window losses
#         is_vae_output: If True, converts from VAE output [-1,1] to [0,255]. If False, assumes input is already in correct format.
    
#     Returns:
#         torch.Tensor or float: Max loss across all windows
#     """
#     from einops import rearrange
#     import numpy as np
    
#     model.eval()
#     num_videos, C, T, H, W = video_tensor.shape
#     device = next(model.parameters()).device
    
#     # Conditional gradient context
#     grad_context = torch.enable_grad() if require_grad else torch.no_grad()
    
#     with grad_context:
#         # Apply same preprocessing as reproduce_intphys_clean.py
#         import sys
#         sys.path.append('/home/yjianhao/project/quentinecode/vjepa2')
#         from app.vjepa.transforms import make_transforms
        
#         transform = make_transforms(
#             random_horizontal_flip=False,
#             random_resize_aspect_ratio=[1/1, 1/1],
#             random_resize_scale=[1.0, 1.0], 
#             reprob=0.,
#             auto_augment=False,
#             motion_shift=False,
#             crop_size=256
#         )
        
#         if is_vae_output:
#             # Debug: Check input tensor properties
#             # print(f"DEBUG: Input video_tensor shape: {video_tensor.shape}")
#             # print(f"DEBUG: Input video_tensor dtype: {video_tensor.dtype}")
#             # print(f"DEBUG: Input video_tensor min: {video_tensor.min():.4f}, max: {video_tensor.max():.4f}")
#             # print(f"DEBUG: Input video_tensor has NaN: {torch.isnan(video_tensor).any()}")
#             # print(f"DEBUG: Input video_tensor has Inf: {torch.isinf(video_tensor).any()}")
            
#             # Convert VAE output [-1,1] to [0,255] like PIL images, then apply V-JEPA transforms
#             # Handle potential NaN/inf values
#             if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
#                 # print("WARNING: Found NaN or Inf values in video_tensor, replacing with zeros")
#                 video_tensor = torch.nan_to_num(video_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
#             # Clamp input to expected range to prevent issues
#             video_tensor = torch.clamp(video_tensor, -1.0, 1.0)
            
#             video_255 = (video_tensor + 1.0) / 2.0 * 255.0
#             # print(f"DEBUG: After scaling to [0,255] - min: {video_255.min():.4f}, max: {video_255.max():.4f}")
            
#             video_frame = video_255.squeeze(0).permute(1, 2, 3, 0).cpu()  # [T, H, W, C] on CPU
#             # print(f"DEBUG: video_frame shape after permute: {video_frame.shape}")
            
#             # Save frames to temporary directory for inspection
#             # temp_dir = "./temp/vjepa_frames"
#             # os.makedirs(temp_dir, exist_ok=True)
#             # print(f"Saving PIL frames to: {temp_dir}")
            
#             # # Convert to uint8 and save each frame
#             # video_uint8 = video_frame.clamp(0, 255).to(torch.uint8).numpy()
#             # print(f"DEBUG: video_uint8 shape: {video_uint8.shape}, dtype: {video_uint8.dtype}")
#             # print(f"DEBUG: video_uint8 min: {video_uint8.min()}, max: {video_uint8.max()}")
            
#             # for frame_idx in range(video_uint8.shape[0]):
#             #     frame = video_uint8[frame_idx]  # [H, W, C]
#             #     print(f"DEBUG: Frame {frame_idx} - shape: {frame.shape}, min: {frame.min()}, max: {frame.max()}")
                
#             #     # Additional check for all-zero frames
#             #     if frame.max() == 0:
#             #         print(f"WARNING: Frame {frame_idx} is completely black (all zeros)!")
                
#             #     pil_image = Image.fromarray(frame)
#             #     frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
#             #     pil_image.save(frame_path)
            
#             # print(f"Saved {video_uint8.shape[0]} frames to {temp_dir}")
            
#             video_normalized = transform(video_frame).unsqueeze(0).to(device)  # [B, C, T, H, W]
#         else:
#             # Input is already in correct format (e.g., from IntPhys dataset after transforms)
#             video_normalized = video_tensor.to(device)
        
#         # Update model parameters exactly as in reproduce_intphys_clean.py
#         model.nb_context_frames = context_length
#         model.frames_per_clip = frames_per_clip
#         model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
        
#         # Create sliding windows exactly as in reproduce_intphys_clean.py  
#         pieces = video_normalized.unfold(2, model.frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
#         pieces = pieces.flatten(0, 1)
#         pieces = rearrange(pieces, "b t c h w -> b c t h w")

        
        
#         # Collect all predictions and targets exactly like reference implementation
#         chunked_preds = []
#         chunked_targets = []
#         CHUNK_SIZE = 1  # Process one at a time for memory efficiency
        
#         with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
#             for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
#                 chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]
                
#                 preds, targets = model(chunk)
#                 chunked_preds.append(preds.cpu())
#                 chunked_targets.append(targets.cpu())
        
#         # Combine all chunks exactly as in reference implementation
#         preds = torch.vstack(chunked_preds)
#         targets = torch.vstack(chunked_targets)
#         preds = preds.view(num_videos, -1, *preds.shape[1:])
#         targets = targets.view(num_videos, -1, *targets.shape[1:])
        
#         # Compute loss exactly as Quentin does
#         loss = F.l1_loss(preds, targets, reduction="none").mean((2, 3))
        
        
        
#         # Recompute with gradients if needed
#         preds_grad = torch.vstack([p.to(device) for p in chunked_preds])
#         targets_grad = torch.vstack([t.to(device) for t in chunked_targets])
#         preds_grad = preds_grad.view(num_videos, -1, *preds_grad.shape[1:])
#         targets_grad = targets_grad.view(num_videos, -1, *targets_grad.shape[1:])
#         # print(f"preds_grad shape: {preds_grad.shape}")
#         # print(f"targets_grad shape: {targets_grad.shape}")
#         # import pdb; pdb.set_trace()
#         loss_grad = F.l1_loss(preds_grad, targets_grad, reduction="none").mean((2, 3))
#         # print(f"loss_grad shape: {loss_grad.shape}")
        
#         if mode == 'max':
#             final_loss = torch.max(loss_grad)
#             # print(f"final_loss: {final_loss}")
#         elif mode == 'mean':
#             final_loss = torch.mean(loss_grad)
        


#         print(f"final_loss: {final_loss}")
        
#         if require_grad:
#             if return_arr:
#                 return final_loss, loss_grad
#             else:
#                 return final_loss
#         else:
#             if return_arr:
#                 return final_loss.detach().item(), loss_grad
#             else:
#                 return final_loss.detach().item()  # Return float

def calculate_torch_vjepa_loss(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=False, mode='max', return_arr=False, is_vae_output=True):
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
    
    Returns:
        torch.Tensor or float: Max loss across all windows
    """
    from einops import rearrange
    import numpy as np
    
    model.eval()
    num_videos, C, T, H, W = video_tensor.shape
    device = next(model.parameters()).device
    
    # Conditional gradient context
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()
    
    with grad_context:
        # Apply same preprocessing as reproduce_intphys_clean.py
        import sys
        sys.path.append('/home/yjianhao/project/quentinecode/vjepa2')
        from app.vjepa.transforms import make_transforms
        
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=[1/1, 1/1],
            random_resize_scale=[1.0, 1.0], 
            reprob=0.,
            auto_augment=False,
            motion_shift=False,
            crop_size=256
        )
        
        if is_vae_output:
            # Convert VAE output [-1,1] to [0,255] like PIL images, then apply V-JEPA transforms
            video_255 = (video_tensor + 1.0) / 2.0 * 255.0
            video_frame = video_255.squeeze(0).permute(1, 2, 3, 0).cpu()  # [T, H, W, C] on CPU
            
            # # Save frames to temporary directory for inspection
            # temp_dir = "./temp"
            # print(f"Saving PIL frames to: {temp_dir}")
            
            # # Convert to uint8 and save each frame
            # video_uint8 = video_frame.clamp(0, 255).to(torch.uint8).numpy()
            # for frame_idx in range(video_uint8.shape[0]):
            #     frame = video_uint8[frame_idx]  # [H, W, C]
            #     pil_image = Image.fromarray(frame)
            #     frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            #     pil_image.save(frame_path)
            
            # print(f"Saved {video_uint8.shape[0]} frames to {temp_dir}")
            
            video_normalized = transform(video_frame).unsqueeze(0).to(device)  # [B, C, T, H, W]
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
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
                chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]
                
                preds, targets = model(chunk)
                chunked_preds.append(preds.cpu())
                chunked_targets.append(targets.cpu())
        
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
        # print(f"preds_grad shape: {preds_grad.shape}")
        # print(f"targets_grad shape: {targets_grad.shape}")
        # import pdb; pdb.set_trace()
        loss_grad = F.l1_loss(preds_grad, targets_grad, reduction="none").mean((2, 3))
        # print(f"loss_grad shape: {loss_grad.shape}")
        
        if mode == 'max':
            final_loss = torch.max(loss_grad)
            # print(f"final_loss: {final_loss}")
        elif mode == 'mean':
            final_loss = torch.mean(loss_grad)
        


        print(f"final_loss: {final_loss}")
        
        if require_grad:
            if return_arr:
                return final_loss, loss_grad
            else:
                return final_loss
        else:
            if return_arr:
                return final_loss.detach().item(), loss_grad
            else:
                return final_loss.detach().item()  # Return float

def calculate_torch_vjepa_loss_v2(video_tensor, model, processor, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True, require_grad=False, mode='max', return_arr=False, is_vae_output=True):
    """
    Calculate V-JEPA loss for PyTorch V-JEPA models (vit_giant_xformers_rope), 
    following the same pattern as calculate_torch_vjepa_loss but adapted for PyTorch implementation.
    Uses sliding windows with proper batching and chunking.
    
    Args:
        video_tensor: Tensor of shape [B, C, T, H, W] or numpy array [T, H, W, C]
        model: PyTorch V-JEPA model (vit_giant_xformers_rope)
        processor: PyTorch video transform pipeline (from build_pt_video_transform)
        context_length: Context length to use 
        frames_per_clip: Number of frames per clip (16 for V-JEPA)
        stride: Stride for sliding window
        use_bfloat16: Whether to use bfloat16 precision
        require_grad: Whether to keep gradients (True) or return detached value (False)
        mode: 'max' or 'mean' for aggregating losses across windows
        return_arr: Whether to return individual window losses
        is_vae_output: If True, converts from VAE output [-1,1] to [0,255]. If False, assumes input is already in correct format.
    
    Returns:
        torch.Tensor or float: Max/mean loss across all windows
        Optional: loss array if return_arr=True
    """
    import numpy as np
    import torch.nn.functional as F
    
    model.eval()
    device = next(model.parameters()).device
    
    # Conditional gradient context
    grad_context = torch.enable_grad() if require_grad else torch.no_grad()
    
    with grad_context:
        # Handle input preprocessing
        if isinstance(video_tensor, np.ndarray):
            # Convert numpy array to tensor
            if video_tensor.ndim == 4:  # [T, H, W, C]
                video_tensor = torch.from_numpy(video_tensor).float().permute(0, 3, 1, 2)  # [T, C, H, W]
                video_tensor = video_tensor.unsqueeze(0)  # [B, T, C, H, W]
                video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        if video_tensor.dim() == 4:  # [T, C, H, W]
            video_tensor = video_tensor.unsqueeze(0)  # [B, T, C, H, W]
            video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        num_videos, C, T, H, W = video_tensor.shape
        
        if is_vae_output:
            # Convert VAE output [-1,1] to [0,255] like PIL images
            if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
                video_tensor = torch.nan_to_num(video_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
            video_tensor = torch.clamp(video_tensor, -1.0, 1.0)
            video_255 = (video_tensor + 1.0) / 2.0 * 255.0
            video_frame = video_255.squeeze(0).permute(1, 2, 3, 0).cpu()  # [T, H, W, C] on CPU
            
            # Apply PyTorch transforms
            video_normalized = processor(video_frame).unsqueeze(0).to(device)  # [B, C, T, H, W]
        else:
            # Input is already in correct format, just ensure it's on the right device
            if video_tensor.device != device:
                video_tensor = video_tensor.to(device)
            video_normalized = video_tensor
        
        # Create sliding windows using unfold (similar to Quentin's approach)
        pieces = video_normalized.unfold(2, frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
        pieces = pieces.flatten(0, 1)  # Flatten batch and window dimensions
        pieces = pieces.permute(0, 2, 1, 3, 4)  # [B*windows, C, T, H, W]
        
        print(f"Video tensor shape: {video_normalized.shape}")
        print(f"Pieces shape: {pieces.shape}")
        print(f"Number of windows: {pieces.shape[0]}")
        
        # Process each window with masking (similar to get_sliding_window_score_torch)
        CHUNK_SIZE = 1  # Process one window at a time for memory efficiency
        loss_arr = []
        
        # Set up masking parameters
        patch_size = 16
        is_mae = False
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
                chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]
                
                # Get masks for this chunk
                m, m_, full_m = get_time_masks(
                    n_timesteps=context_length, 
                    spatial_size=(patch_size, patch_size), 
                    temporal_dim=frames_per_clip, 
                    as_bool=is_mae
                )
                
                full_m = full_m.unsqueeze(0).to(device)
                m = m.unsqueeze(0).to(device)
                m_ = m_.unsqueeze(0).to(device)
                
                # Get features from PyTorch V-JEPA model
                h = model(chunk)  # [batch, num_patches, feature_dim]
                
                # Normalize features
                h = F.layer_norm(h, (h.size(-1),))
                
                # Apply masks to get target patches
                masks_pred = [m_.repeat(chunk.shape[0], 1)]
                targets = apply_masks(h, masks_pred, concat=False)
                
                if len(targets) > 0:
                    target_features = targets[0]  # [num_masked_patches, feature_dim]
                    
                    # Compute L1 loss for this window (following V-JEPA pattern)
                    window_loss = torch.abs(target_features).mean()
                    
                    if require_grad:
                        loss_arr.append(window_loss)
                    else:
                        loss_arr.append(window_loss.detach())
                    
                    print(f"Window {chunk_id}: loss = {window_loss.item():.6f}")
                else:
                    print(f"Window {chunk_id}: No valid masks, skipping")
        
        if len(loss_arr) == 0:
            print("Warning: No valid losses computed")
            if require_grad:
                return torch.tensor(0.0, device=device, requires_grad=True)
            else:
                if return_arr:
                    return 0.0, []
                else:
                    return 0.0
        
        # Aggregate losses following the same pattern as calculate_torch_vjepa_loss
        if require_grad:
            loss_tensor = torch.stack(loss_arr)
        else:
            loss_tensor = torch.stack(loss_arr)
        
        # Create gradient-enabled version for final computation
        loss_grad = torch.stack([l.to(device) if not l.requires_grad else l for l in loss_arr])
        
        if mode == 'max':
            final_loss = torch.max(loss_grad)
        elif mode == 'mean':
            final_loss = torch.mean(loss_grad)
        
        print(f"Final loss ({mode}): {final_loss}")
        
        if require_grad:
            if return_arr:
                return final_loss, loss_grad
            else:
                return final_loss
        else:
            if return_arr:
                return final_loss.detach().item(), loss_grad.detach()
            else:
                return final_loss.detach().item()

