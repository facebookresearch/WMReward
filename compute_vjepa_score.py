import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
import numpy as np
import torch.nn.functional as F

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

def apply_masks(x, masks, concat=True):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)

def get_score(video,model,processor):
    video = processor(video, return_tensors="pt").to(model.device)
    model.eval()

    device = model.device
    patch_size = 16
    frames_per_clip = 32
    is_mae = False
    num_videos = 1
    B = num_videos
    n_timesteps = 2
    m, m_, full_m = get_time_masks(n_timesteps=n_timesteps, spatial_size=(patch_size, patch_size), temporal_dim=frames_per_clip, as_bool=is_mae)

    full_m = full_m.unsqueeze(0).to(device)
    m = m.unsqueeze(0).to(device)
    m_ = m_.unsqueeze(0).to(device)

    if is_mae:
        masks_enc = m.repeat(B, 1)
        masks_pred = m_.repeat(B, 1)
        full_mask = full_m.repeat(B, 1)
    else:
        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]
        full_mask = [full_m.repeat(B, 1)]

    h = model(**video, skip_predictor=True).last_hidden_state
    targets = apply_masks(h, masks_pred, concat=False)

    outputs = model(**video, context_mask=masks_enc, target_mask=masks_pred)
    preds = outputs.predictor_output.last_hidden_state

    preds = preds[0].view(num_videos, -1, *preds[0].shape[1:])
    targets = targets[0].view(num_videos, -1, *targets[0].shape[1:]).squeeze(0)
    loss = F.l1_loss(preds, targets, reduction="none").mean().detach()
    
    return loss.item()

# Example usage
# video_url = "/home/yjianhao/project/video_guidance/temp/wan-t2v.mp4"
# score = process_video(video_url)
# print(f"Video Score: {score}")