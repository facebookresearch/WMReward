#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class IntPhysDataset(Dataset):
    """
    IntPhys Dataset following the original implementation structure.
    
    Expected data structure:
    data_path/ (e.g., O1, O2, O3)
    ├── 01/
    │   ├── 1/
    │   │   ├── scene/
    │   │   │   ├── frame_000.jpg
    │   │   │   └── ...
    │   │   └── status.json
    │   ├── 2/ (similar structure)
    │   ├── 3/ (similar structure)
    │   └── 4/ (similar structure)
    ├── 02/ (similar structure)
    └── ...
    """
    
    def __init__(
        self,
        data_path,
        frames_per_clip=16,
        frame_step=4,
        transform=None,
        return_paths=False
    ):
        """
        Args:
            data_path: Path to IntPhys property (e.g., /path/to/IntPhys/dev/O1/)
            frames_per_clip: Number of frames to sample per clip
            frame_step: Step size between sampled frames
            transform: Transform to apply to frames
            return_paths: Whether to return file paths in output
        """
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.transform = transform
        self.return_paths = return_paths
        
        # Get all scene directories (like 01, 02, 03, etc.)
        self.scenes = sorted([d for d in os.listdir(self.data_path) 
                             if os.path.isdir(os.path.join(self.data_path, d))])
        
        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found in {self.data_path}")
        
        self.length_clip = self.frames_per_clip * self.frame_step
        
        # Default transform if none provided - matches original implementation
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

    def _load_frames(self, scene, possibility):
        """Load frames for a specific scene and possibility following original structure."""
        frames_dir = os.path.join(self.data_path, scene, str(possibility), "scene")
        
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        # Get all frame files and sort them
        frames_all = sorted([f for f in os.listdir(frames_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(frames_all) < self.length_clip:
            # If not enough frames, start from beginning
            start = 0
        else:
            # Random start position like original
            start = np.random.randint(0, len(frames_all) - self.length_clip)
            
        # Handle edge case for very long clips
        if self.length_clip > 90:
            start = 0
            
        # Sample frames with step like original
        frames_to_load = frames_all[start:start + self.length_clip:self.frame_step]
        
        # Load frames as PIL images first
        clip = []
        for frame in frames_to_load:
            frame_path = os.path.join(frames_dir, frame)
            frame_ = Image.open(frame_path).convert('RGB')
            clip.append(frame_)
        
        # Apply transforms to PIL images (transforms handle PIL -> Tensor conversion)
        if self.transform is not None:
            clip = [self.transform(frame) for frame in clip]
            clip = torch.stack(clip)
        else:
            # Manual conversion if no transform
            clip = [torch.tensor(np.array(frame)).permute(2, 0, 1).float() / 255.0 for frame in clip]
            clip = torch.stack(clip)
            
        return clip

    def _load_label(self, scene, possibility):
        """Load label from status.json following original implementation."""
        status_path = os.path.join(self.data_path, scene, str(possibility), "status.json")
        
        try:
            with open(status_path, 'r') as f:
                a = json.load(f)
            label = a['header']['is_possible']
        except:
            label = True  # Default like original
            
        return label

    def __getitem__(self, index):
        """Get a scene with all 4 possibilities following original implementation."""
        scene = self.scenes[index]
        
        labels = []
        buffer = []
        paths = []
        
        # Load all 4 possibilities for this scene like original
        for possibility in [1, 2, 3, 4]:
            try:
                # Load video clip
                clip = self._load_frames(scene, possibility)
                buffer.append(clip)
                
                # Load label
                label = self._load_label(scene, possibility)
                labels.append(label)
                
                # Store path if requested (matches original format)
                if self.return_paths:
                    paths.append(f"{self.data_path[-2:]}/{scene}/{possibility}")
                    
            except Exception as e:
                print(f"Warning: Error loading {scene}/{possibility}: {e}")
                # Create dummy data for missing possibilities
                dummy_clip = torch.zeros(self.frames_per_clip, 3, 224, 224)
                buffer.append(dummy_clip)
                labels.append(False)
                if self.return_paths:
                    paths.append(f"{self.data_path[-2:]}/{scene}/{possibility}")
        
        # Stack clips and convert labels like original
        buffer = torch.stack(buffer)  # Shape: (4, frames, 3, H, W)
        labels = torch.Tensor(labels)
        id = torch.Tensor([index])
        
        if self.return_paths:
            return buffer, labels, paths
        else:
            return buffer, labels, id

    def __len__(self):
        return len(self.scenes)


def create_intphys_dataloader(
    data_path,
    batch_size=1,
    frames_per_clip=16,
    frame_step=4,
    transform=None,
    num_workers=4,
    shuffle=False,
    return_paths=False
):
    """
    Create a DataLoader for IntPhys dataset following original implementation.
    
    Args:
        data_path: Path to IntPhys property (e.g., /path/to/IntPhys/dev/O1/)
        batch_size: Batch size
        frames_per_clip: Number of frames per clip
        frame_step: Frame sampling step
        transform: Image transform
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        return_paths: Whether to return file paths
    
    Returns:
        DataLoader instance
    """
    dataset = IntPhysDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        transform=transform,
        return_paths=return_paths
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    
    # You need to specify the actual path to your IntPhys dataset
    # For example: "/path/to/IntPhys/dev/O1/"
    data_path = "/home/yjianhao/project/video_guidance/dev/O3"  # Update this path
    
    try:
        # Create dataloader
        dataloader = create_intphys_dataloader(
            data_path=data_path,
            batch_size=2,
            frames_per_clip=8,
            frame_step=2,
            num_workers=2,
            shuffle=True,
            return_paths=True
        )
        
        print(f"Dataset created with {len(dataloader.dataset)} scenes")
        print(f"Dataloader has {len(dataloader)} batches")
        
        # Test loading one batch
        for batch_idx, (clips, labels, paths) in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Clips shape: {clips.shape}")  # Should be (batch_size, 4, frames, 3, H, W)
            print(f"  Labels shape: {labels.shape}")  # Should be (batch_size, 4)
            print(f"  Labels: {labels}")
            print(f"  Paths: {paths}")
            
            # Only test first batch
            break
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the data_path variable to point to your IntPhys dataset")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
