import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
from torchvision import transforms

# Custom Dataset
class F1LapDataset(Dataset):
    def __init__(self, image_data, metrics_data, lap_times):
        self.images = image_data  # Shape: (N, C, H, W)
        self.metrics = metrics_data  # Shape: (N, num_metrics)
        self.lap_times = lap_times  # Shape: (N, 1)

    def __len__(self):
        return len(self.lap_times)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'metrics': self.metrics[idx],
            'lap_time': self.lap_times[idx]
        }

# Hybrid Model
class HybridModel(nn.Module):
    def __init__(self, num_metrics):
        super().__init__()
        # CNN Branch (using EfficientNet for images)
        self.cnn = models.efficientnet_b0(pretrained=True)
        self.cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Grayscale input
        self.cnn.classifier = nn.Identity()  # Remove final layer
        
        # Tabular Branch (for circuit metrics)
        self.mlp = nn.Sequential(
            nn.Linear(num_metrics, 64),
            nn.ReLU(),
            nn.Linear(64, 32))
        
        # Combined Head
        self.head = nn.Sequential(
            nn.Linear(1280 + 32, 512),  # EfficientNet-B0 out_dim = 1280
            nn.ReLU(),
            nn.Linear(512, 1))
    def forward(self, image, metrics):
        # CNN
        image_features = self.cnn(image)  # (batch, 1280)
        # Tabular
        metric_features = self.mlp(metrics)  # (batch, 32)
        # Concatenate
        combined = torch.cat([image_features, metric_features], dim=1)
        return self.head(combined)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(num_metrics=10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()  # Mixed precision
criterion = nn.MSELoss()

# DataLoader (example)
dataset = F1LapDataset(image_data, metrics_data, lap_times)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training Loop
for epoch in range(50):
    for batch in dataloader:
        images = batch['image'].to(device)
        metrics = batch['metrics'].to(device)
        targets = batch['lap_time'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():  # Mixed precision
            outputs = model(images, metrics)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')