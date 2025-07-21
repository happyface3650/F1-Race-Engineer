from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder

images_dir = "track_images//png"

def get_mean(images_dir):
    dataset = ImageFolder(images_dir)
    pixels = np.stack([np.array(img) for img, _ in dataset]) / 255.0
    mean = pixels.mean(axis=(0, 1, 2))  # [R_mean, G_mean, B_mean]
    std = pixels.std(axis=(0, 1, 2))
    print("Mean:", mean, "Std:", std)

#Mean: [0.03727689 0.05267702 0.03467816] Std: [0.1552852  0.18150549 0.14338638]
