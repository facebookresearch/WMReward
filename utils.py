import torch
import torchvision.transforms as transforms
import sys
import numpy as np
import importlib


def preprocess_video_for_torch_vjepa(frames):
    """
    Convert list of PIL Images to tensor format for torch V-JEPA model.
    Uses the exact same transforms as Quentin's V-JEPA2 implementation.
    
    Args:
        frames: List of PIL Images from video generation
        
    Returns:
        video_tensor: Tensor of shape [B, C, T, H, W] ready for torch V-JEPA
    """
    sys.path.append('/home/yjianhao/project/quentinecode/vjepa2')
    from app.vjepa.transforms import make_transforms
    
    # Use exact same transforms as in clean reproduction / Quentin's code
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1/1, 1/1],  # No aspect ratio change
        random_resize_scale=[1.0, 1.0],  # No scale change  
        reprob=0.,
        auto_augment=False,  # This makes normalization use 255.0 scale
        motion_shift=False,
        crop_size=256  # V-JEPA2 uses 256x256, not 224x224!
    )
    
    # Convert PIL Images to tensor in format expected by V-JEPA2 transforms
    # V-JEPA2 transform expects: [T, H, W, C] format
    
    # Convert PIL to numpy arrays
    frame_arrays = []
    for frame in frames:
        frame_array = np.array(frame)
        if len(frame_array.shape) == 2:  # Grayscale
            frame_array = np.stack([frame_array] * 3, axis=-1)  # Convert to RGB
        elif frame_array.shape[-1] == 4:  # RGBA
            frame_array = frame_array[..., :3]  # Remove alpha channel
        frame_arrays.append(frame_array)
    
    # Stack to [T, H, W, C] format
    video_array = np.stack(frame_arrays, axis=0)
    video_tensor = torch.tensor(video_array, dtype=torch.float32)
    
    # Apply V-JEPA2 transform (this handles normalization correctly)
    video_tensor = transform(video_tensor)  # Returns [C, T, H, W]
    
    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0)  # [B, C, T, H, W]
    
    return video_tensor


def init_torch_vjepa():
    """Initialize Quentin's torch V-JEPA model using config from video_guidance/config/"""
    
    # Add Quentin's codebase to path
    sys.path.insert(0, '/home/yjianhao/project/quentinecode/vjepa2')
    
    import yaml
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration from YAML file
    config_path = '/home/yjianhao/project/video_guidance/config/vjepa_2_h_open.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model configuration
    model_kwargs = config['model_kwargs']['pretrain_kwargs']
    wrapper_kwargs = config['model_kwargs']['wrapper_kwargs']
    checkpoint_path = config['model_kwargs']['checkpoint']
    module_name = config['model_kwargs']['module_name']
    
    # Get experiment data parameters
    experiment_data = config['experiment']['data']
    frames_per_clip = experiment_data['frames_per_clip']
    
    # Import and initialize model exactly as in reproduce_intphys_clean.py
    model = importlib.import_module(module_name).init_module(
        frames_per_clip=frames_per_clip,
        nb_context_frames=1,  # Will be updated dynamically
        checkpoint=checkpoint_path,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)

    
    print(f"Torch V-JEPA model loaded from {checkpoint_path} using config {config_path}")
    return model 