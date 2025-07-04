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
import cv2

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

def preprocess_video_for_analysis(video_path, target_size=(256, 256), target_frames=None, remove_black_edges=True):
    """
    Preprocess video to handle different resolutions, frame rates, and black edges
    """
    # Load video with opencv for better control
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    
    # Remove black edges if needed
    if remove_black_edges:
        frames = remove_black_borders(frames)
    
    # Resize frames
    processed_frames = []
    for frame in frames:
        # Convert to PIL for resizing
        pil_frame = Image.fromarray(frame)
        resized_frame = pil_frame.resize(target_size, Image.Resampling.LANCZOS)
        processed_frames.append(np.array(resized_frame))
    
    # Handle frame rate differences by resampling to target number of frames
    if target_frames and len(processed_frames) != target_frames:
        processed_frames = resample_frames(processed_frames, target_frames)
    
    return processed_frames

def remove_black_borders(frames, threshold=10):
    """
    Remove black borders from video frames
    """
    if len(frames) == 0:
        return frames
    
    # Use first frame to detect borders
    first_frame = frames[0]
    h, w = first_frame.shape[:2]
    
    # Find non-black regions (pixels above threshold)
    gray = np.mean(first_frame, axis=2) if len(first_frame.shape) == 3 else first_frame
    non_black = gray > threshold
    
    # Find bounding box of non-black region
    rows = np.any(non_black, axis=1)
    cols = np.any(non_black, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # If all black, return original
        return frames
    
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    # Crop all frames to remove black borders
    cropped_frames = []
    for frame in frames:
        cropped = frame[top:bottom+1, left:right+1]
        cropped_frames.append(cropped)
    
    return cropped_frames

def resample_frames(frames, target_frames):
    """
    Resample frames to target number using linear interpolation
    """
    current_frames = len(frames)
    if current_frames == target_frames:
        return frames
    
    # Create indices for resampling
    indices = np.linspace(0, current_frames - 1, target_frames)
    resampled_frames = []
    
    for idx in indices:
        if idx == int(idx):
            # Exact frame
            resampled_frames.append(frames[int(idx)])
        else:
            # Interpolate between two frames
            idx1, idx2 = int(np.floor(idx)), int(np.ceil(idx))
            weight = idx - idx1
            
            frame1 = frames[idx1].astype(np.float32)
            frame2 = frames[idx2].astype(np.float32)
            interpolated = ((1 - weight) * frame1 + weight * frame2).astype(np.uint8)
            resampled_frames.append(interpolated)
    
    return resampled_frames

def create_video_strip(video_np, num_frames_to_show=8):
    """Create a horizontal strip of video frames"""
    total_frames = len(video_np)
    if num_frames_to_show >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        # Select evenly spaced frames
        frame_indices = np.linspace(0, total_frames-1, num_frames_to_show, dtype=int)
    
    # Get selected frames and concatenate horizontally
    selected_frames = [video_np[i] for i in frame_indices]
    video_strip = np.concatenate(selected_frames, axis=1)  # Concatenate along width
    
    return video_strip, frame_indices

def create_comprehensive_plot(video_data_list, model_name, context_length, save_path):
    """
    Create a comprehensive plot showing multiple videos and their loss curves
    video_data_list: list of tuples (video_name, video_np, loss_arr, score)
    """
    num_videos = len(video_data_list)
    
    # Create figure with video strips on top and loss curves below
    fig = plt.figure(figsize=(16, 4 + 3 * num_videos))
    
    # Create grid: video strips take 2/3 height, loss plots take 1/3
    gs = fig.add_gridspec(num_videos + 1, 1, height_ratios=[2] * num_videos + [3])
    
    # Colors for different videos
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot video strips
    for i, (video_name, video_np, loss_arr, score) in enumerate(video_data_list):
        ax_video = fig.add_subplot(gs[i, 0])
        
        # Create video strip
        video_strip, frame_indices = create_video_strip(video_np, num_frames_to_show=8)
        
        ax_video.imshow(video_strip)
        ax_video.set_title(f'{video_name} (Score: {score:.4f})', fontsize=12, fontweight='bold')
        ax_video.set_xticks([])
        ax_video.set_yticks([])
        
        # Add frame numbers
        frame_width = video_strip.shape[1] // len(frame_indices)
        for j, frame_idx in enumerate(frame_indices):
            x_pos = (j + 0.5) * frame_width
            ax_video.text(x_pos, video_strip.shape[0] + 10, f'F{frame_idx}', 
                         ha='center', va='bottom', fontsize=8)
    
    # Plot all loss curves together
    ax_loss = fig.add_subplot(gs[-1, 0])
    
    for i, (video_name, video_np, loss_arr, score) in enumerate(video_data_list):
        # Convert loss_arr to numpy if needed
        if torch.is_tensor(loss_arr):
            loss_arr_np = loss_arr.cpu().numpy().flatten()
        else:
            loss_arr_np = np.array(loss_arr)
        
        color = colors[i % len(colors)]
        ax_loss.plot(loss_arr_np, color=color, linewidth=2, 
                    label=f'{video_name} (Score: {score:.4f})', marker='o', markersize=3)
    
    ax_loss.set_title(f'PyTorch V-JEPA Loss Comparison ({model_name.upper()}, Context={context_length})', 
                     fontsize=14, fontweight='bold')
    ax_loss.set_xlabel('Timestep', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add model info
    info_text = f'Model: {model_name.upper()}\nContext Length: {context_length}\nTotal Videos: {num_videos}'
    ax_loss.text(0.02, 0.98, info_text, 
                transform=ax_loss.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive plot saved: {save_path}")

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

# Define models and context lengths to test
models_to_test = {
    'vitg': "/home/yjianhao/project/vjepa2/checkpoints/vitg-384.pt",
    'vith': "/home/yjianhao/project/quentinecode/vjepa2/vit-h-open/vith.pt",
    # Add more PyTorch models here if available
    # 'vitg-384': "/home/yjianhao/project/vjepa2/checkpoints/vitg-384.pt",
}

context_lengths_to_test = [2, 4, 6, 8, 10]

# Initialize common parameters
img_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test groups
test_groups = {
    'pyramid_videos': {
        'videos': [
            "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/valid_00.mp4", 
            "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/temporal_disorder_00.mp4",
            "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_phase_shifting_00.mp4",
            "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_sphere_fusion_00.mp4",
            "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_teleporting_spheres_00.mp4"
        ],
        'use_diffusers_loader': True,  # Use existing load_video function
        'target_frames': None
    },
    'guidance_videos': {
        'videos': [
            "/home/yjianhao/project/video_guidance/base.mp4",
            "/home/yjianhao/project/video_guidance/generated.mp4"
        ],
        'use_diffusers_loader': False,  # Use custom preprocessing
        'target_frames': 64  # Target number of frames
    }
}

# Loop through test groups, models and context lengths
for group_name, group_config in test_groups.items():
    print(f"\n{'='*80}")
    print(f"TESTING GROUP: {group_name.upper()}")
    print(f"Videos: {len(group_config['videos'])}")
    print(f"{'='*80}")
    
    for model_name, model_path in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Testing PyTorch model: {model_name} on {group_name}")
        print(f"Checkpoint: {model_path}")
        print(f"{'='*60}")
        

        model = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
        model.to(device).eval()
        
        # Load pretrained weights
        load_pretrained_vjepa_pt_weights(model, model_path)
        
        # Build PyTorch preprocessing transform
        processor = build_pt_video_transform(img_size=img_size)
        
        print(f"Successfully loaded {model_name}")

        
        for context_length in context_lengths_to_test:
            print(f"\n--- Testing {model_name} with context_length={context_length} on {group_name} ---")
            
            # Create organized output directory
            dir_name = f"./debug/pyramid_comparison/{group_name}/{model_name}_context{context_length}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            
            # Collect data for comprehensive plot
            video_data_list = []
            
            for raw_path in group_config['videos']:
                try:
                    # Load video based on group configuration
                    if group_config['use_diffusers_loader']:
                        # Use existing diffusers loader for pyramid videos
                        video = load_video(raw_path)
                        video_np = np.stack([np.array(frame.resize((256, 256))) for frame in video], axis=0)
                    else:
                        # Use custom preprocessing for guidance videos
                        video_np = preprocess_video_for_analysis(
                            raw_path, 
                            target_size=(256, 256),
                            target_frames=group_config['target_frames'],
                            remove_black_edges=True
                        )
                        video_np = np.array(video_np)
                    
                    print(f"Video shape: {video_np.shape}")

                    # Use the clean PyTorch-native function
                    score, loss_arr = get_sliding_window_score_torch(
                        video_np, model, processor, 
                        kernel_size=16, 
                        context_window_size=context_length,  # Use the current context length
                        stride=2, 
                        return_form='arr'
                    )
                    print(f"Score: {score}")

                    # Get video name for processing
                    video_name = raw_path.split('/')[-1].split('.')[0]
                    
                    # Add to comprehensive plot data
                    video_data_list.append((video_name, video_np, loss_arr, score))
                    
                    # Create individual video strip for visualization
                    video_strip, frame_indices = create_video_strip(video_np, num_frames_to_show=8)
                    
                    # Create individual subplot figure
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
                    
                    # Plot video frames on top
                    ax1.imshow(video_strip)
                    ax1.set_title(f'Video Frames: {video_name} ({group_name})')
                    ax1.set_xlabel('Frame Sequence')
                    ax1.set_ylabel('Height')
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    
                    # Add frame numbers as labels
                    frame_width = video_strip.shape[1] // len(frame_indices)
                    for i, frame_idx in enumerate(frame_indices):
                        x_pos = (i + 0.5) * frame_width
                        ax1.text(x_pos, video_strip.shape[0] + 10, f'F{frame_idx}', 
                                ha='center', va='bottom', fontsize=8)
                    
                    # Plot loss curve on bottom
                    ax2.plot(loss_arr, 'b-', linewidth=2)
                    ax2.set_title(f'PyTorch V-JEPA Loss Over Time ({model_name.upper()}, Context={context_length}, {group_name})')
                    ax2.set_xlabel('Timestep')
                    ax2.set_ylabel('Loss')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add comprehensive annotation
                    info_text = f'Model: {model_name.upper()}\nContext: {context_length}\nGroup: {group_name}\nScore: {score:.4f}'
                    ax2.text(0.02, 0.98, info_text, 
                             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    
                    plt.tight_layout()
                    
                    # Save the individual combined plot
                    plot_filename = f"combined_{model_name}_ctx{context_length}_{video_name}.png"
                    plt.savefig(f"{dir_name}/{plot_filename}", 
                                dpi=150, bbox_inches='tight')
                    plt.close()  # Close to free memory
                    
                    print(f"Saved individual: {plot_filename}")
                    
                except Exception as e:
                    video_name = raw_path.split('/')[-1].split('.')[0]
                    print(f"Error processing {video_name} with {model_name} context={context_length}: {e}")
                    continue
            
            # Generate comprehensive plot for all videos in this model/context combination
            if video_data_list:
                comprehensive_filename = f"COMPREHENSIVE_{model_name}_ctx{context_length}_{group_name}_all_videos.png"
                comprehensive_path = f"{dir_name}/{comprehensive_filename}"
                create_comprehensive_plot(video_data_list, f"{model_name} ({group_name})", context_length, comprehensive_path)
            else:
                print(f"No valid data for comprehensive plot: {model_name} context={context_length} {group_name}")

print(f"\n{'='*60}")
print("PyTorch testing complete! Results saved in ./debug/pyramid_comparison/")
print("Folder structure: {group_name}/{model_name}_context{context_length}/")
print("Test groups: pyramid_videos, guidance_videos")
print(f"{'='*60}")