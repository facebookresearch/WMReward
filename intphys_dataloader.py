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
    Minimal IntPhys Dataset for loading intuitive physics videos.
    
    Expected data structure:
    data_path/
    ├── scene_001/
    │   ├── 1/
    │   │   ├── scene/
    │   │   │   ├── frame_000.jpg
    │   │   │   └── ...
    │   │   └── status.json
    │   ├── 2/ (similar structure)
    │   ├── 3/ (similar structure)
    │   └── 4/ (similar structure)
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
            data_path: Path to IntPhys dataset
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
        
        # Get all scene directories
        self.scenes = sorted([d for d in os.listdir(self.data_path) 
                             if os.path.isdir(os.path.join(self.data_path, d))])
        
        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found in {self.data_path}")
        
        self.length_clip = self.frames_per_clip * self.frame_step
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

    def _load_frames(self, scene_path, possibility):
        """Load frames for a specific scene and possibility."""
        frames_dir = os.path.join(scene_path, str(possibility), "scene")
        
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(frame_files) < self.length_clip:
            # If not enough frames, start from beginning
            start = 0
        else:
            # Random start position
            start = np.random.randint(0, len(frame_files) - self.length_clip + 1)
        
        # Sample frames with step
        frames_to_load = frame_files[start:start + self.length_clip:self.frame_step]
        
        # Load and transform frames
        clip = []
        for frame_file in frames_to_load:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            clip.append(frame)
        
        # Apply transforms
        if self.transform:
            clip = [self.transform(frame) for frame in clip]
        
        return torch.stack(clip)

    def _load_label(self, scene_path, possibility):
        """Load label from status.json."""
        status_path = os.path.join(scene_path, str(possibility), "status.json")
        
        try:
            with open(status_path, 'r') as f:
                status_data = json.load(f)
            return status_data['header']['is_possible']
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            # Default to True if status file is missing or malformed
            return True

    def __getitem__(self, index):
        """Get a scene with all 4 possibilities."""
        scene = self.scenes[index]
        scene_path = os.path.join(self.data_path, scene)
        
        clips = []
        labels = []
        paths = []
        
        # Load all 4 possibilities for this scene
        for possibility in [1, 2, 3, 4]:
            try:
                # Load video clip
                clip = self._load_frames(scene_path, possibility)
                clips.append(clip)
                
                # Load label
                label = self._load_label(scene_path, possibility)
                labels.append(label)
                
                # Store path if requested
                if self.return_paths:
                    paths.append(f"{scene}/{possibility}")
                    
            except Exception as e:
                print(f"Warning: Error loading {scene}/{possibility}: {e}")
                # Create dummy data for missing possibilities
                dummy_clip = torch.zeros(self.frames_per_clip, 3, 224, 224)
                clips.append(dummy_clip)
                labels.append(False)
                if self.return_paths:
                    paths.append(f"{scene}/{possibility}")
        
        # Stack clips and convert labels
        clips = torch.stack(clips)  # Shape: (4, frames, 3, H, W)
        labels = torch.tensor(labels, dtype=torch.bool)
        
        if self.return_paths:
            return clips, labels, paths
        else:
            return clips, labels

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
    Create a DataLoader for IntPhys dataset.
    
    Args:
        data_path: Path to IntPhys dataset
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
        pin_memory=True
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
