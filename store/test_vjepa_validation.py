#!/usr/bin/env python3

"""
Quick validation script to test if the modified test_vjepa_loss_pipeline.py works correctly
"""

import os
import sys
import torch
import logging

# Add Quentin's codebase to path
sys.path.insert(0, '/home/yjianhao/project/quentinecode/vjepa2')

# Import our modules
from test_vjepa_loss_pipeline import *
from compute_vjepa_score import calculate_torch_vjepa_loss

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def test_single_scene():
    """Test the pipeline on a single scene to validate it works"""
    
    data_path = "/home/yjianhao/project/video_guidance/dev/O1"
    checkpoint_path = "/home/yjianhao/project/quentinecode/vjepa2/vit-h-open/vith.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize transform exactly as Quentin does
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
    
    # Initialize model
    model = init_model(checkpoint_path, device)
    
    # Initialize dataset
    dataset = IntPhysDataset(
        data_path=data_path,
        frames_per_clip=99//2,
        frame_step=2,
        transform=transform
    )
    
    # Test on first scene only
    logger.info(f"Testing on first scene: {dataset.scenes[0]}")
    scene_data = dataset[0]
    
    # Check data format
    videos, labels, ids = scene_data
    logger.info(f"Videos shape: {videos.shape}")
    logger.info(f"Video value range: [{videos.min():.6f}, {videos.max():.6f}]")
    logger.info(f"Labels: {labels}")
    
    # Test our function on a single video
    single_video = videos[0:1].to(device)  # First video only
    logger.info(f"Testing calculate_torch_vjepa_loss on single video: {single_video.shape}")
    
    # Test with correct parameters
    try:
        loss_val, window_losses = calculate_torch_vjepa_loss(
            single_video,
            model,
            context_length=2,
            frames_per_clip=16,
            stride=2,
            use_bfloat16=True,
            require_grad=False,
            mode='max',
            return_arr=True,
            is_vae_output=False  # Key parameter for IntPhys
        )
        
        logger.info(f"Loss value: {loss_val}")
        logger.info(f"Window losses shape: {window_losses.shape}")
        logger.info(f"Window losses: {window_losses}")
        logger.info("✓ calculate_torch_vjepa_loss works correctly!")
        
    except Exception as e:
        logger.error(f"✗ Error in calculate_torch_vjepa_loss: {e}")
        raise
    
    # Test the full scene extraction
    try:
        scene_losses, scene_labels = extract_losses_for_scene_using_our_function(
            model, scene_data, context_lengths=[2], 
            frames_per_clip=16, stride=2, use_bfloat16=True, device=device
        )
        
        logger.info(f"Scene losses length: {len(scene_losses)}")
        logger.info(f"Scene labels length: {len(scene_labels)}")
        if len(scene_losses) > 0:
            logger.info(f"First pair losses shape: {scene_losses[0].shape}")
            logger.info(f"First pair labels: {scene_labels[0]}")
        
        logger.info("✓ extract_losses_for_scene_using_our_function works correctly!")
        
    except Exception as e:
        logger.error(f"✗ Error in extract_losses_for_scene_using_our_function: {e}")
        raise
    
    logger.info("🎉 All tests passed! The pipeline is working correctly.")

if __name__ == "__main__":
    test_single_scene() 