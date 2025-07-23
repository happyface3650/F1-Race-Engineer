import pandas as pd
import torch.nn as nn
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from CNN import TrackCNN

class Model(nn.Module):
    def __init__(self, num_tab, num_aux, num_classes=1,
                 image_embed_size=256, tabular_embed_size=128, num_heads=8, 
                 num_layers=3, dropout=0.1):
        super(Model, self).__init__()
        self.track_cnn = TrackCNN(output_dim=image_embed_size)

        self.image_adaptor = nn.Linear(64, image_embed_size)
        




