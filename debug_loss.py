
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video
import torch
from torchcodec.decoders import VideoDecoder
import numpy as np
import torch.nn.functional as F
# from compute_vjepa_score import get_sliding_window_score_max

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

def get_sliding_window_score_max(video, model, processor, kernel_size, context_window_size, stride=2):
    video = processor(video, return_tensors="pt").to(model.device)
    # print(processor)

    print(f"Video shape after processing: {video['pixel_values_videos'].shape}")
    B, T, C, H, W = video['pixel_values_videos'].shape
    model.eval()
    device = model.device
    patch_size = 16
    temporal_size = 2
    is_mae=False
    frames_per_clip = T
    spatial_dim = (H, W) 
    num_videos = B
    B = num_videos
    start_index_arr = np.arange(0, frames_per_clip - kernel_size + 1, stride)

    loss_arr = []
    for start_index in start_index_arr:
        m, m_, full_m = get_sequential_mask(spatial_size=(patch_size, patch_size), temporal_size=temporal_size, spatial_dim=spatial_dim, temporal_dim=frames_per_clip, start_frame=start_index, kernel_size=kernel_size, context_window_size=context_window_size, as_bool=False)

        full_m = full_m.unsqueeze(0).to(device)
        m = m.unsqueeze(0).to(device)
        m_ = m_.unsqueeze(0).to(device)
        print(f"Mask shapes: m={m.shape}, m_={m_.shape}, full_m={full_m.shape}")

        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]

        h = model(**video, skip_predictor=True).last_hidden_state
        targets = apply_masks(h, masks_pred, concat=False)

        outputs = model(**video, context_mask=masks_enc, target_mask=masks_pred)
        preds = outputs.predictor_output.last_hidden_state

        preds = preds[0].view(num_videos, -1, *preds[0].shape[1:])
        targets = targets[0].view(num_videos, -1, *targets[0].shape[1:]).squeeze(0)
        loss = F.l1_loss(preds, targets, reduction="none").mean().detach()
        print(f"Start index: {start_index}, Loss: {loss.item()}")  # Debugging line to inspect loss values
        loss_arr.append(loss.item())
    
    # loss_avr = np.mean(loss_arr)
    loss_avr = np.max(loss_arr)  # Use max instead of mean

    return loss_avr

# init vjepa
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, model.config.frames_per_clip, 2) # you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width
print(f"Video shape: {video.shape}")
# import pdb; pdb.set_trace()


score = get_sliding_window_score_max(video, model, processor, kernel_size=8, context_window_size=4, stride=2)
print(f"Score: {score}")
