import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video, load_video
import matplotlib.pyplot as plt
from compute_vjepa_score import get_score, get_sliding_window_score, get_sliding_window_score_based
from PIL import Image
import shutil

# Add the vjepa2 project root to path (not just src)
sys.path.append('/home/yjianhao/project/vjepa2')

# Import the PyTorch implementation components
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    if not os.path.exists(pretrained_weights):
        print(f"Warning: Checkpoint not found at {pretrained_weights}")
        return
    
    try:
        pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    except Exception as e:
        print(f"Error loading weights: {e}")

def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return eval_transform

def get_sliding_window_score_torch(video, model, processor, kernel_size, context_window_size, stride=2, return_form='arr', mode='max', require_grad=False):
    """
    PyTorch-native version of sliding window score function
    """
    # Import the masking functions from compute_vjepa_score
    from compute_vjepa_score import get_time_masks, apply_masks
    
    # Process video with PyTorch transforms
    if isinstance(video, list):
        video_np = np.stack([np.array(frame) for frame in video], axis=0)
    else:
        video_np = video
    
    # Convert to tensor and apply transforms
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # T x C x H x W
    video_tensor = processor(video_tensor)  # Apply transforms
    
    # Get device from model parameters
    device = next(model.parameters()).device
    video_tensor = video_tensor.unsqueeze(0).to(device)  # 1 x T x C x H x W
    
    model.eval()
    B, C, T, H, W = video_tensor.shape  # This is actually B x C x T x H x W format
    
    # For sliding window, we work with T dimension (temporal dimension is index 2)
    patch_size = 16
    is_mae = False
    spatial_dim = (H, W)
    start_index_arr = np.arange(0, T - kernel_size + 1, stride)
    
    print(f"Video tensor shape: {video_tensor.shape}")
    print(f"Start indices: {start_index_arr}")
    print(f"Number of windows: {len(start_index_arr)}")

    loss_arr = []
    for i, start_index in enumerate(start_index_arr):
        try:
            # Slice along temporal dimension (dimension 2): B x C x T x H x W
            video_slice = video_tensor[:, :, start_index:start_index+kernel_size]  # 1 x C x kernel_size x H x W
            print(f"Window {i}: slice shape {video_slice.shape}")
            
            # Already in correct B x C x T x H x W format for the model
            print(f"Window {i}: model input shape {video_slice.shape}")
            
            # Get masks
            m, m_, full_m = get_time_masks(n_timesteps=context_window_size, spatial_size=(patch_size, patch_size), temporal_dim=kernel_size, as_bool=is_mae)
            
            full_m = full_m.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)
            m_ = m_.unsqueeze(0).to(device)

            masks_enc = [m.repeat(B, 1)]
            masks_pred = [m_.repeat(B, 1)]

            # Get features from model
            h = model(video_slice)  # Direct PyTorch model call
            print(f"Model output shape: {h.shape}")
            
            # Normalize targets
            normalize_targets = True
            if normalize_targets:
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
            
            # Simplified approach: use predictor mask to get target patches
            # and compute a simple reconstruction loss
            targets = apply_masks(h, masks_pred, concat=False)
            
            if len(targets) > 0:
                target_features = targets[0]  # [num_masked_patches, feature_dim]
                print(f"Window {i}: target features shape: {target_features.shape}")
                
                # Use L1 loss (mean absolute value)
                loss = torch.abs(target_features).mean()
                
                if require_grad:
                    loss_arr.append(loss)
                else:
                    loss_arr.append(loss.detach().item())
                    
                print(f"Window {i}: loss = {loss.item():.6f} (L1 loss)")
            else:
                print(f"Window {i}: No valid masks, skipping")
                
        except Exception as e:
            print(f"Error in window {i}: {e}")
            continue

    if require_grad:
        loss_tensor = torch.stack(loss_arr)
        if mode == 'max':
            final_loss = torch.max(loss_tensor)
        elif mode == 'mean':
            final_loss = torch.mean(loss_tensor)
        return final_loss
    else:
        # Original logic for non-gradient case
        if mode == 'max':
            final_loss = np.max(loss_arr)
        elif mode == 'mean':
            final_loss = np.mean(loss_arr)

        if return_form == 'arr':
            return final_loss, loss_arr
        else:
            return final_loss

# Initialize PyTorch VJEPA model
img_size = 256  # Use 256 like in the original debug_loss.py
pt_model_path = "/home/yjianhao/project/vjepa2/checkpoints/vitg-256.pt"  # Adjust path as needed

model = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
model.cuda().eval()

# Load pretrained weights (you'll need to download/have the checkpoint)
load_pretrained_vjepa_pt_weights(model, pt_model_path)

# Build PyTorch preprocessing transform
processor = build_pt_video_transform(img_size=img_size)

raw_paths = ["/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/valid_00.mp4", 
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/temporal_disorder_00.mp4",
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_phase_shifting_00.mp4",
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_sphere_fusion_00.mp4",
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_teleporting_spheres_00.mp4"]

for raw_path in raw_paths:
    video = load_video(raw_path)

    # video is a list of PIL.Image.Image objects
    video_np = np.stack([np.array(frame.resize((256, 256))) for frame in video], axis=0)
    print(video_np.shape)

    # Use the clean PyTorch-native function
    score, loss_arr = get_sliding_window_score_torch(video, model, processor, kernel_size=16, context_window_size=10, stride=2, return_form='arr')
    print(f"Score: {score}")

    dir_name = "./debug/pyramid2_pytorch"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    
    save_path = f"{dir_name}/{raw_path.split('/')[-1].split('.')[0]}.mp4"
    shutil.copy2(raw_path, save_path)
    plt.figure(figsize=(6,6))
    plt.plot(loss_arr)

    # Set title and labels
    plt.xlabel('timestep')
    plt.ylabel('Loss')
    # Save the plot locally
    plt.savefig(f"{dir_name}/loss_{raw_path.split('/')[-1].split('.')[0]}.png")
    plt.close()  # Close the figure to free memory