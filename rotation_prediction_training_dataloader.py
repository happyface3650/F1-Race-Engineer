import numpy as np
from torchvision.transforms import RandomRotation
from torch.utils.data import Dataset, DataLoader
import torch

from dataset import ImageTabularDataset

train_data = ImageTabularDataset("track_images\\png", use_aux=False )

class RotationDataset(Dataset):
    def __init__(self, image_tensors):
        """
        Args:
            image_tensors: List of [3, H, W] track image tensors
        """
        self.images = image_tensors  # Your original track images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        angle = np.random.choice([0, 90, 180, 270])  # Random rotation
        rotated_img = RandomRotation([angle, angle])(img)  # Apply rotation
        label = angle // 90  # Convert to class (0-3)
        return rotated_img, label

# Example usage:
track_images =   train_data
rotation_dataset = RotationDataset(track_images)
rotation_dataloader = DataLoader(rotation_dataset, batch_size=32, shuffle=True)