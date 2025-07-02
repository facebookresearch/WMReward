import torch
# from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
import numpy as np
import torch.nn.functional as F
from diffusers.utils import export_to_video, load_video

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

def get_unified_vjepa_loss(video_frames, model, processor, kernel_size, context_window_size, stride=2, mode='mean', model_type='hf'):
    """
    Unified V-JEPA loss computation that works with both HuggingFace and Quentin's models.
    
    Args:
        video_frames: List of PIL images or numpy array
        model: Either HuggingFace V-JEPA model or Quentin's JEPA model
        processor: Processor function (for HF) or transform function (for Quentin)
        kernel_size: Size of the sliding window
        context_window_size: Size of the context within the window
        stride: Stride for sliding window
        mode: 'mean' or 'max' for aggregating across windows
        model_type: 'hf' for HuggingFace model, 'quentin' for Quentin's model
    
    Returns:
        loss: Computed loss value
    """
    if model_type == 'hf':
        # Use existing HuggingFace implementation
        return get_sliding_window_score_based(
            video=video_frames,
            model=model,
            processor=processor,
            kernel_size=kernel_size,
            context_window_size=context_window_size,
            stride=stride,
            return_form='loss',
            mode=mode,
            require_grad=False
        )
    
    elif model_type == 'quentin':
        # Implementation for Quentin's model
        # Convert video_frames to the format expected by Quentin's model
        if isinstance(video_frames, list):
            # Convert PIL images to numpy array
            import numpy as np
            from PIL import Image
            video_np = np.array([np.array(frame) for frame in video_frames])
        else:
            video_np = video_frames
        
        # Apply transform if it's a function (Quentin's transform)
        if callable(processor):
            video_tensor = processor(video_np)  # Should return (3, T, H, W)
        else:
            # Assume it's already a tensor
            video_tensor = torch.tensor(video_np).permute(3, 0, 1, 2)  # (3, T, H, W)
        
        # Add batch dimension and move to device
        # Handle different model types that may not have direct device attribute
        try:
            device = model.device
        except AttributeError:
            # Try to get device from model parameters
            device = next(model.parameters()).device
        
        video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, 3, T, H, W)
        B, C, T, H, W = video_tensor.shape
        
        # Sliding window approach
        start_indices = np.arange(0, T - kernel_size + 1, stride)
        loss_arr = []
        
        for start_idx in start_indices:
            # Extract window
            window = video_tensor[:, :, start_idx:start_idx + kernel_size]  # (1, 3, kernel_size, H, W)
            
            try:
                # Update model parameters for this context length
                old_nb_context = getattr(model, 'nb_context_frames', None)
                old_frames_per_clip = getattr(model, 'frames_per_clip', None)
                
                model.nb_context_frames = context_window_size
                model.frames_per_clip = kernel_size
                
                # Try to update grid_depth
                try:
                    model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
                except:
                    try:
                        model.grid_depth = model.frames_per_clip // model.encoder.backbone.tubelet_size
                    except:
                        model.grid_depth = model.frames_per_clip // 2
                
                # Forward pass through Quentin's model
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    preds, targets = model(window)
                    
                    # Compute L1 loss
                    loss = F.l1_loss(preds, targets, reduction="none").mean()
                    loss_arr.append(loss.detach().cpu().item())
                
                # Restore old parameters
                if old_nb_context is not None:
                    model.nb_context_frames = old_nb_context
                if old_frames_per_clip is not None:
                    model.frames_per_clip = old_frames_per_clip
                    
            except Exception as e:
                print(f"Error in Quentin model forward pass: {e}")
                loss_arr.append(0.0)
        
        # Aggregate losses
        if not loss_arr:
            return 0.0
            
        if mode == 'max':
            return np.max(loss_arr)
        elif mode == 'mean':
            return np.mean(loss_arr)
        else:
            return np.mean(loss_arr)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'hf' or 'quentin'")

def calculate_torch_vjepa_loss(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True):
    """
    Calculate V-JEPA loss using Quentin's torch implementation.
    Uses the same approach as reproduce_intphys_clean.py
    
    Args:
        video_tensor: Tensor of shape [B, C, T, H, W] where B is number of videos (e.g., 4 for IntPhys)
        model: Quentin's V-JEPA model (loaded via init_module)
        context_length: Context length to use (single value)
        frames_per_clip: Number of frames per clip for sliding window
        stride: Stride for sliding window
        use_bfloat16: Whether to use bfloat16 precision
    
    Returns:
        float: Mean loss value
    """
    from einops import rearrange
    import numpy as np
    
    model.eval()
    device = video_tensor.device
    num_videos = video_tensor.shape[0]
    
    with torch.no_grad():
        # Update model parameters exactly as in reproduce_intphys_clean.py
        model.nb_context_frames = context_length
        model.frames_per_clip = frames_per_clip
        
        # Set grid_depth exactly as in reproduce_intphys_clean.py
        model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
        
        # Create sliding windows exactly as in reproduce_intphys_clean.py
        pieces = video_tensor.unfold(2, model.frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
        pieces = pieces.flatten(0, 1)
        pieces = rearrange(pieces, "b t c h w -> b c t h w")
        
        # Process in chunks exactly as in reproduce_intphys_clean.py
        chunked_preds = []
        chunked_targets = []
        CHUNK_SIZE = 2  # Same as reproduce_intphys_clean.py
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
                chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]
                
                preds, targets = model(chunk)
                chunked_preds.append(preds.cpu())
                chunked_targets.append(targets.cpu())
        
        preds = torch.vstack(chunked_preds)
        targets = torch.vstack(chunked_targets)
        preds = preds.view(num_videos, -1, *preds.shape[1:])
        targets = targets.view(num_videos, -1, *targets.shape[1:])
        
        # Compute loss exactly as in reproduce_intphys_clean.py
        loss = F.l1_loss(preds, targets, reduction="none").mean((2, 3)).detach().to(device)
        
        # Return mean loss
        return loss.mean().item()

def calculate_torch_vjepa_loss_with_grad(video_tensor, model, context_length=4, frames_per_clip=16, stride=2, use_bfloat16=True):
    """
    Calculate V-JEPA loss using Quentin's torch implementation with gradient support.
    This version enables gradients for use in guidance pipelines.
    Uses the same approach as reproduce_intphys_clean.py
    
    Args:
        video_tensor: Tensor of shape [B, C, T, H, W] where B is number of videos
        model: Quentin's V-JEPA model (loaded via init_module)
        context_length: Context length to use (single value)
        frames_per_clip: Number of frames per clip for sliding window
        stride: Stride for sliding window
        use_bfloat16: Whether to use bfloat16 precision
    
    Returns:
        torch.Tensor: Loss tensor with gradients enabled
    """
    from einops import rearrange
    import numpy as np
    
    model.eval()
    device = video_tensor.device
    num_videos = video_tensor.shape[0]
    
    # Update model parameters exactly as in reproduce_intphys_clean.py
    model.nb_context_frames = context_length
    model.frames_per_clip = frames_per_clip
    
    # Set grid_depth exactly as in reproduce_intphys_clean.py
    model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
    
    # Create sliding windows exactly as in reproduce_intphys_clean.py
    pieces = video_tensor.unfold(2, model.frames_per_clip, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
    pieces = pieces.flatten(0, 1)
    pieces = rearrange(pieces, "b t c h w -> b c t h w")
    
    # Process in chunks exactly as in reproduce_intphys_clean.py - but keep gradients
    chunked_preds = []
    chunked_targets = []
    CHUNK_SIZE = 2  # Same as reproduce_intphys_clean.py
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
        for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
            chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]
            
            preds, targets = model(chunk)
            chunked_preds.append(preds)  # Keep on device with gradients
            chunked_targets.append(targets)  # Keep on device with gradients
    
    preds = torch.vstack(chunked_preds)
    targets = torch.vstack(chunked_targets)
    preds = preds.view(num_videos, -1, *preds.shape[1:])
    targets = targets.view(num_videos, -1, *targets.shape[1:])
    
    # Compute loss exactly as in reproduce_intphys_clean.py - but keep gradients
    loss = F.l1_loss(preds, targets, reduction="none").mean((2, 3))
    
    # Return mean loss with gradients
    return loss.mean()

