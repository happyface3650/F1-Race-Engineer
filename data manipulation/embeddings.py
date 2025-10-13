import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from dataset import ImageTabularDataset

device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu')

def create_track_embeddings_dict(tracks_dir):
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    track_embeddings = {}

    for img_file in os.listdir(tracks_dir):
        if img_file.endswith('.png'):
            track_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(tracks_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                features = model(image_tensor)
                embedding = features.cpu().numpy().flatten()

            track_embeddings[track_name] = embedding
    return track_embeddings



