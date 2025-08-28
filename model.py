import torch
import torch.nn as nn

class F1LapTimePredictor(nn.Module):

    def __init__(self, num_numeric_features)