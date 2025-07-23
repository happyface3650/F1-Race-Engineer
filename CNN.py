
import torch.nn as nn

class TrackCNN(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        # Input: 3-channel color-coded diagram (e.g., DRS zones, sectors)
        self.features = nn.Sequential(
            # Stage 1: Detect basic color zones
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # [8, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),  # Halve resolution
            
            # Stage 2: Extract spatial relationships
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # [16, H/2, W/2]
            nn.ReLU(),
            
            # Stage 3: Final feature aggregation
            nn.Conv2d(16, 32, kernel_size=1),  # [32, H/2, W/2] (1x1 conv)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # [32, 1, 1]
        )
        
        self.projection = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = self.features(x)  # [batch, 32, 1, 1]
        x = x.flatten(1)  # [batch, 32]
        return self.projection(x)  # [batch, 256]