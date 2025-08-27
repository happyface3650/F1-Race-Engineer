import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from IFCNN.CNN import TrackCNN

model = TrackCNN()


