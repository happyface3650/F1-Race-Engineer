import matplotlib.pyplot as plt
from CNN import TrackCNN
import torch
from torch import nn
from dataset import ImageTabularDataset
import torch.nn.functional as F
from torchvision import transforms
data = ImageTabularDataset( 'All_Laps_Train.csv', 'track_images\\png\\train\\tracks', 256, True)

image, tabular, aux, target1, target2 = data[1]

model = TrackCNN(output_dim=256)

def generate_cam(model, img_tensor):
    """Optimized for 2D color-coded track diagrams"""
    # Hook the last convolutional layer
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    # Register hook - using the conv layer before AdaptiveAvgPool
    hook = model.features[-2].register_forward_hook(hook_fn)
    
    # Forward pass (ensure 4D input)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Get weights and transpose [output_dim, channels] -> [channels, output_dim]
    weights = model.projection.weight.data.T  # [32, 256]
    
    # Select first output dimension (simplest for diagrams)
    weights = weights[:, 0:1]  # [32, 1]
    
    # Compute CAM
    activations = activations[0]  # [1, 32, H, W]
    cam = torch.einsum('ck,kij->ij', weights, activations.squeeze(0))  # Note the 'ij' output
    cam = nn.functional.relu(cam)
    
    # Resize to match input image
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                       size=img_tensor.shape[-2:],
                       mode='bilinear').squeeze()
    
    hook.remove()
    return cam

# Generate and plot
cam = generate_cam(model, image)

def overlay_heatmap(original_img, heatmap):
    """Optimized for color-coded diagrams"""
    # Convert to numpy
    img_np = original_img.squeeze().permute(1, 2, 0).cpu().numpy()
    heatmap_np = heatmap.cpu().numpy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original diagram
    ax1.imshow(img_np)
    ax1.set_title("Original Diagram")
    ax1.axis('off')
    
    # Heatmap
    im = ax2.imshow(heatmap_np, cmap='viridis')
    ax2.set_title("Model Attention")
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    plt.show()

overlay_heatmap(image, cam)

