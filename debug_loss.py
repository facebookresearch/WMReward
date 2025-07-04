from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video, load_video
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from compute_vjepa_score import get_score, get_sliding_window_score,  get_sliding_window_score_based, calculate_torch_vjepa_loss
from PIL import Image
import os
import shutil
import sys
import importlib
import cv2

# Add Quentin's codebase to path
sys.path.insert(0, '/home/yjianhao/project/quentinecode/vjepa2')

# Import Quentin's modules
from app.vjepa.transforms import make_transforms

def init_model(checkpoint_path, device, resolution=256):
    """Initialize model exactly as Quentin does"""
    
    # Model configuration from Quentin's config
    model_kwargs = {
        'resolution': resolution,
        'encoder': {
            'model_name': 'vit_huge',
            'checkpoint_key': 'encoder',
            'is_causal': False,
            'local_window': [-1, -1, -1],
            'uniform_power': True,
            'use_activation_checkpointing': True,
            'use_mask_tokens': True,
            'use_rope': True,
            'zero_init_mask_tokens': True,
            'num_frames': 16,
        },
        'target_encoder': {
            'checkpoint_key': 'target_encoder',
        },
        'predictor': {
            'model_name': 'vit_predictor',
            'checkpoint_key': 'predictor',
            'depth': 12,
            'is_causal': False,
            'local_window': [-1, -1, -1],
            'num_heads': 12,
            'uniform_power': True,
            'use_activation_checkpointing': True,
            'use_mask_tokens': True,
            'use_rope': True,
            'zero_init_mask_tokens': True,
            'num_mask_tokens': 10,
            'num_frames': 16,
        }
    }
    
    wrapper_kwargs = {
        'no_predictor': False,
    }
    
    # Import and initialize model exactly as Quentin does
    module_name = 'app.vjepa.modelcustom.vit_encoder_predictor_noar_targets'
    model = importlib.import_module(module_name).init_module(
        frames_per_clip=16,
        nb_context_frames=1,  # Will be updated dynamically
        checkpoint=checkpoint_path,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    return model

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
    
    print(f"Original frames: {len(frames)}, shape: {frames[0].shape}, range: [{np.array(frames).min()}, {np.array(frames).max()}]")
    
    # Remove black edges if needed
    if remove_black_edges:
        frames = remove_black_borders(frames)
        print(f"After black border removal: shape: {frames[0].shape}, range: [{np.array(frames).min()}, {np.array(frames).max()}]")
    
    # Resize frames
    processed_frames = []
    for frame in frames:
        # Convert to PIL for resizing
        pil_frame = Image.fromarray(frame)
        resized_frame = pil_frame.resize(target_size, Image.Resampling.LANCZOS)
        processed_frames.append(np.array(resized_frame))
    
    print(f"After resizing: shape: {processed_frames[0].shape}, range: [{np.array(processed_frames).min()}, {np.array(processed_frames).max()}]")
    
    # Handle frame rate differences by resampling to target number of frames
    if target_frames and len(processed_frames) != target_frames:
        # Extract video name from path for special handling
        video_name = os.path.basename(video_path).split('.')[0]
        processed_frames = resample_frames(processed_frames, target_frames, video_name)
        print(f"After frame resampling: {len(processed_frames)} frames, range: [{np.array(processed_frames).min()}, {np.array(processed_frames).max()}]")
    
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

def resample_frames(frames, target_frames, video_name=None):
    """
    Resample frames to target number using different strategies:
    - For base videos: start from frame 24, take next 64 frames
    - For other videos: take consecutive frames starting from first frame
    """
    current_frames = len(frames)
    
    # Special handling for base video - start from frame 24
    if video_name and 'base' in video_name.lower():
        start_frame = 24
        if current_frames >= start_frame + target_frames:
            # Take target_frames starting from frame 24
            return frames[start_frame:start_frame + target_frames]
        elif current_frames > start_frame:
            # Take what we can from frame 24 onwards, pad with last frame
            available_frames = frames[start_frame:]
            padding_needed = target_frames - len(available_frames)
            padded_frames = available_frames + [available_frames[-1]] * padding_needed
            return padded_frames
        else:
            # Not enough frames to start from frame 24, pad entire sequence
            padding_needed = target_frames
            padded_frames = [frames[-1]] * padding_needed if frames else [np.zeros((256, 256, 3), dtype=np.uint8)] * padding_needed
            return padded_frames
    
    # For other videos: take consecutive frames from start
    if current_frames >= target_frames:
        # Take first target_frames
        return frames[:target_frames]
    else:
        # Pad with last frame if shorter than target
        padding_needed = target_frames - current_frames
        padded_frames = frames + [frames[-1]] * padding_needed
        return padded_frames

def adjust_brightness_contrast(frames, brightness=0, contrast=1.0):
    """
    Adjust brightness and contrast of video frames
    brightness: additive factor (-100 to 100, typically)
    contrast: multiplicative factor (0.5 to 2.0, typically)
    """
    adjusted_frames = []
    for frame in frames:
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)
        
        # Apply contrast (multiplicative) then brightness (additive)
        adjusted = frame_float * contrast + brightness
        
        # Clip to valid range and convert back to uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        adjusted_frames.append(adjusted)
    
    return adjusted_frames

def create_brightness_contrast_variations(base_video_path, output_dir, target_size=(256, 256), target_frames=64):
    """
    Create multiple brightness/contrast variations of the base video
    Returns list of (variation_name, processed_frames) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base video
    base_frames = preprocess_video_for_analysis(
        base_video_path, 
        target_size=target_size,
        target_frames=target_frames,
        remove_black_edges=True
    )
    
    # Define brightness/contrast variations
    variations = [
        ("original", 0, 1.0),
        ("bright_low_contrast", 30, 0.7),
        ("dark_high_contrast", -25, 1.4),
        ("very_bright", 50, 1.0),
        ("very_dark", -40, 1.0),
    ]
    
    variation_data = []
    
    for var_name, brightness, contrast in variations:
        print(f"Creating variation: {var_name} (brightness={brightness}, contrast={contrast})")
        
        # Apply brightness/contrast adjustment
        if var_name == "original":
            adjusted_frames = base_frames
        else:
            adjusted_frames = adjust_brightness_contrast(base_frames, brightness, contrast)
        
        # Save variation as video file for debugging
        variation_path = os.path.join(output_dir, f"base_{var_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 8
        height, width = adjusted_frames[0].shape[:2]
        out = cv2.VideoWriter(variation_path, fourcc, fps, (width, height))
        
        for frame in adjusted_frames:
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Saved variation: {variation_path}")
        
        variation_data.append((var_name, adjusted_frames))
    
    return variation_data

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

def debug_frame_comparison(video_path, output_dir="./debug_frames"):
    """Compare original vs processed frames to debug brightness changes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original with CV2
    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    cap.release()
    
    if ret:
        # Original processing
        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
        # Full preprocessing
        processed_frames = preprocess_video_for_analysis(video_path, target_size=(256, 256), target_frames=None, remove_black_edges=True)
        processed_frame = processed_frames[0]
        
        # Save comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(original_rgb)
        ax1.set_title(f'Original\nRange: [{original_rgb.min()}, {original_rgb.max()}]\nMean: {original_rgb.mean():.1f}')
        ax1.axis('off')
        
        ax2.imshow(processed_frame)
        ax2.set_title(f'Processed\nRange: [{processed_frame.min()}, {processed_frame.max()}]\nMean: {processed_frame.mean():.1f}')
        ax2.axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"comparison_{os.path.basename(video_path)}.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Frame comparison saved: {comparison_path}")
        return comparison_path
    
    return None

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
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
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
    
    ax_loss.set_title(f'V-JEPA Loss Comparison ({model_name.upper()}, Context={context_length})', 
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

# Transform will be initialized per model with appropriate crop_size

# Define models and context lengths to test
models_to_test = {
    'vith': {
        'checkpoint': "/home/yjianhao/project/quentinecode/vjepa2/vit-h-open/vith.pt",
        'resolution': 256
    },
    # 'vitg': {
    #     'checkpoint': "/home/yjianhao/project/vjepa2/checkpoints/vitg-384.pt",
    #     'resolution': 384
    # }
}

context_lengths_to_test = [2, 4, 6, 8, 10]

# Define test groups
test_groups = {
    # 'ball_videos': {
    #     'videos': [
    #         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_005/valid_00.mp4",
    #         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_005/temporal_disorder_00.mp4",
    #         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_005/invalid_phantom_force_00.mp4",
    #         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_005/invalid_penetration_00.mp4",
    #     ],
    #     'use_diffusers_loader': True,  # Use existing load_video function
    #     'target_frames': None  # Keep original frame count
    # },
    # 'pyramid_videos': {
    #     'videos': [
    #         "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/valid_00.mp4", 
    #         "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/temporal_disorder_00.mp4",
    #         "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_phase_shifting_00.mp4",
    #         "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_sphere_fusion_00.mp4",
    #         "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_teleporting_spheres_00.mp4"
    #     ],
    #     'use_diffusers_loader': True,  # Use existing load_video function
    #     'target_frames': None
    # },
    # 'guidance_videos': {
    #     'videos': [
    #         "/home/yjianhao/project/video_guidance/base.mp4",
    #         "/home/yjianhao/project/video_guidance/output_cosmos_24.mp4"
    #     ],
    #     'use_diffusers_loader': False,  # Use custom preprocessing
    #     'target_frames': 64  # Both videos get 64 frames (base starts from frame 24, others from frame 0)
    # },
    # 'brightness_contrast_variations': {
    #     'base_video': "/home/yjianhao/project/video_guidance/base.mp4",
    #     'use_diffusers_loader': False,  # Use custom preprocessing
    #     'target_frames': 64,  # Base video gets 64 frames starting from frame 24
    #     'generate_variations': True
    # },
    "guidance_test": {
        'videos': [
            "/home/yjianhao/project/video_guidance/generated_videos/subject_consistency/guidance_f33_s50_w16c8_rho1.0_cfg5.0torch/a bicycle accelerating to gain speed.mp4",
            "/home/yjianhao/project/video_guidance/generated_videos/subject_consistency/guidance_f33_s50_w16c8_rho1.0_cfg5.0torch/a cow bending down to drink water from a river.mp4",
            "/home/yjianhao/project/video_guidance/generated_videos/subject_consistency/guidance_f33_s50_w16c8_rho1.0_cfg5.0torch/a motorcycle gliding through a snowy field.mp4",
            "/home/yjianhao/project/video_guidance/generated_videos/subject_consistency/guidance_f33_s50_w16c8_rho1.0_cfg5.0torch/a train accelerating to gain speed.mp4"
        ],
        'use_diffusers_loader': False,  # Use custom preprocessing
        'target_frames': 33,  # Base video gets 64 frames starting from frame 24
    }
}

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loop through test groups, models and context lengths
for group_name, group_config in test_groups.items():
    print(f"\n{'='*80}")
    print(f"TESTING GROUP: {group_name.upper()}")
    
    # Handle different group structures
    if 'videos' in group_config:
        print(f"Videos: {len(group_config['videos'])}")
    elif 'base_video' in group_config:
        print(f"Base video: {group_config['base_video']}")
        print(f"Generating variations: {group_config.get('generate_variations', False)}")
    
    print(f"{'='*80}")
    
    for model_name, model_info in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name} on {group_name}")
        print(f"Checkpoint: {model_info['checkpoint']}")
        print(f"Resolution: {model_info['resolution']}")
        print(f"{'='*60}")
        
        # Initialize model for this checkpoint
        try:
            model = init_model(model_info['checkpoint'], device, model_info['resolution'])
            
            # Initialize transform with model-specific resolution
            transform = make_transforms(
                random_horizontal_flip=False,
                random_resize_aspect_ratio=[1/1, 1/1],
                random_resize_scale=[1.0, 1.0], 
                reprob=0.,
                auto_augment=False,
                motion_shift=False,
                crop_size=model_info['resolution']
            )
            
            print(f"Successfully loaded {model_name} with {model_info['resolution']}x{model_info['resolution']} resolution")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
        
        for context_length in context_lengths_to_test:
            print(f"\n--- Testing {model_name} with context_length={context_length} on {group_name} ---")
            
            # Create organized output directory
            dir_name = f"./debug/pyramid2_comparison/{group_name}/{model_name}_context{context_length}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            
            # Collect data for comprehensive plot
            video_data_list = []
            
            # Handle brightness/contrast variations group specially
            if group_name == 'brightness_contrast_variations':
                # Generate brightness/contrast variations
                variations_dir = f"./debug_frames/{group_name}/variations"
                variation_data = create_brightness_contrast_variations(
                    group_config['base_video'], 
                    variations_dir,
                    target_size=(model_info['resolution'], model_info['resolution']),
                    target_frames=group_config['target_frames']
                )
                
                # Process each variation
                for video_name, video_np in variation_data:
                    video_np = np.array(video_np)
                    
                    print(f"Processing variation: {video_name}")
                    print(f"Video shape: {video_np.shape}")
                    print(f"Video dtype: {video_np.dtype}")
                    print(f"Video value range: [{video_np.min():.2f}, {video_np.max():.2f}]")
                    print(f"Video mean: {video_np.mean():.2f}")
                    
                    # Apply Quentin's transforms to match the expected input format
                    video_tensor = torch.from_numpy(video_np).float()  # [T, H, W, C]
                    video_transformed = transform(video_tensor).unsqueeze(0).to(device)  # [B, C, T, H, W]

                    # Use the new loss function with Quentin's model
                    score, loss_arr = calculate_torch_vjepa_loss(
                        video_tensor=video_transformed,
                        model=model,
                        context_length=context_length,  # Use the current context length
                        frames_per_clip=16,  # equivalent to kernel_size
                        stride=2,
                        use_bfloat16=True,
                        require_grad=False,
                        mode='max',
                        return_arr=True,
                        is_vae_output=False  # Using Quentin's transforms, input is in correct format
                    )
                    print(f"Score: {score}")

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
                    
                    # Convert loss_arr to numpy if it's a tensor
                    if torch.is_tensor(loss_arr):
                        loss_arr_np = loss_arr.cpu().numpy().flatten()
                    else:
                        loss_arr_np = loss_arr
                    
                    # Plot loss curve on bottom
                    ax2.plot(loss_arr_np, 'r-', linewidth=2)
                    ax2.set_title(f'V-JEPA Loss Over Time ({model_name.upper()}, Context={context_length}, {group_name})')
                    ax2.set_xlabel('Timestep')
                    ax2.set_ylabel('Loss')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add comprehensive annotation
                    info_text = f'Model: {model_name.upper()}\nContext: {context_length}\nGroup: {group_name}\nScore: {score:.4f}'
                    ax2.text(0.02, 0.98, info_text, 
                             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    plt.tight_layout()
                    
                    # Save the individual combined plot
                    plot_filename = f"combined_{model_name}_ctx{context_length}_{video_name}.png"
                    plt.savefig(f"{dir_name}/{plot_filename}", 
                                dpi=150, bbox_inches='tight')
                    plt.close()  # Close to free memory
                    
                    print(f"Saved individual: {plot_filename}")
            
            else:
                # Handle regular video groups
                for raw_path in group_config['videos']:
                    # Get video name for processing
                    video_name = raw_path.split('/')[-1].split('.')[0]
                    
                    # Debug: Create frame comparison for guidance videos
                    if not group_config['use_diffusers_loader']:
                        debug_frame_comparison(raw_path, f"./debug_frames/{group_name}")
                    
                    # Load video based on group configuration
                    if group_config['use_diffusers_loader']:
                        # Use existing diffusers loader for pyramid videos
                        video = load_video(raw_path)
                        video_np = np.stack([np.array(frame.resize((model_info['resolution'], model_info['resolution']))) for frame in video], axis=0)
                    else:
                        # Use custom preprocessing for guidance videos
                        video_np = preprocess_video_for_analysis(
                            raw_path, 
                            target_size=(model_info['resolution'], model_info['resolution']),
                            target_frames=group_config['target_frames'],
                            remove_black_edges=True
                        )
                        video_np = np.array(video_np)
                        
                        # Save processed video for guidance videos
                        processed_video_dir = f"./debug_frames/{group_name}/processed_videos"
                        os.makedirs(processed_video_dir, exist_ok=True)
                        processed_video_path = f"{processed_video_dir}/{video_name}_processed_{model_info['resolution']}p.mp4"
                        
                        # Save processed video using OpenCV
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = 8  # Reasonable FPS for debug videos
                        height, width = video_np.shape[1:3]
                        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
                        
                        for frame in video_np:
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                        
                        out.release()
                        print(f"Saved processed video: {processed_video_path}")
                    
                    print(f"Video shape: {video_np.shape}")
                    print(f"Video dtype: {video_np.dtype}")
                    print(f"Video value range: [{video_np.min():.2f}, {video_np.max():.2f}]")
                    print(f"Video mean: {video_np.mean():.2f}")
                    
                    # Apply Quentin's transforms to match the expected input format
                    video_tensor = torch.from_numpy(video_np).float()  # [T, H, W, C]
                    video_transformed = transform(video_tensor).unsqueeze(0).to(device)  # [B, C, T, H, W]

                    # Use the new loss function with Quentin's model
                    score, loss_arr = calculate_torch_vjepa_loss(
                        video_tensor=video_transformed,
                        model=model,
                        context_length=context_length,  # Use the current context length
                        frames_per_clip=16,  # equivalent to kernel_size
                        stride=2,
                        use_bfloat16=True,
                        require_grad=False,
                        mode='max',
                        return_arr=True,
                        is_vae_output=False  # Using Quentin's transforms, input is in correct format
                    )
                    print(f"Score: {score}")

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
                    
                    # Convert loss_arr to numpy if it's a tensor
                    if torch.is_tensor(loss_arr):
                        loss_arr_np = loss_arr.cpu().numpy().flatten()
                    else:
                        loss_arr_np = loss_arr
                    
                    # Plot loss curve on bottom
                    ax2.plot(loss_arr_np, 'r-', linewidth=2)
                    ax2.set_title(f'V-JEPA Loss Over Time ({model_name.upper()}, Context={context_length}, {group_name})')
                    ax2.set_xlabel('Timestep')
                    ax2.set_ylabel('Loss')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add comprehensive annotation
                    info_text = f'Model: {model_name.upper()}\nContext: {context_length}\nGroup: {group_name}\nScore: {score:.4f}'
                    ax2.text(0.02, 0.98, info_text, 
                             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    plt.tight_layout()
                    
                    # Save the individual combined plot
                    plot_filename = f"combined_{model_name}_ctx{context_length}_{video_name}.png"
                    plt.savefig(f"{dir_name}/{plot_filename}", 
                                dpi=150, bbox_inches='tight')
                    plt.close()  # Close to free memory
                    
                    print(f"Saved individual: {plot_filename}")
            
            # Generate comprehensive plot for all videos in this model/context combination
            if video_data_list:
                comprehensive_filename = f"COMPREHENSIVE_{model_name}_ctx{context_length}_{group_name}_all_videos.png"
                comprehensive_path = f"{dir_name}/{comprehensive_filename}"
                create_comprehensive_plot(video_data_list, f"{model_name} ({group_name})", context_length, comprehensive_path)
            else:
                print(f"No valid data for comprehensive plot: {model_name} context={context_length} {group_name}")

print(f"\n{'='*60}")
print("Quentin's V-JEPA testing complete! Results saved in ./debug/pyramid2_comparison/")
print("Folder structure: {group_name}/{model_name}_context{context_length}/")
print("Test groups: ball_videos, guidance_videos, brightness_contrast_variations")
print("Brightness/contrast variations: original, bright_low_contrast, dark_high_contrast, very_bright, very_dark")
print(f"{'='*60}")
