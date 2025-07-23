import torch.nn as nn
import torch
import pandas as pd
class PositionEncoder(nn.Module):
    def __init__(self, max_pos=20):
        super().__init__()
        # Linear scaling for ordinality
        self.position_scale = 1.0 / max_pos
        
        # Embedding for non-linear effects
        self.embedding = nn.Embedding(max_pos, 8)  # +2 for DNF/DQ
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        


    def forward(self, positions):
        # Input can be tensor or pandas series
        if isinstance(positions, pd.Series):
            positions = positions.fillna("").astype(str)
        
        # Convert to numerical
        numeric_pos = torch.zeros(len(positions), dtype=torch.long)
        for i, pos in enumerate(positions):
            numeric_pos[i] = min(int(float(pos)), self.embedding.num_embeddings - 2) + 1
        
        # Get both representations
        scaled = numeric_pos.float() * self.position_scale  # Linear [0,1]
        embedded = self.embedding(numeric_pos)  # Non-linear
        
        return torch.cat([scaled.unsqueeze(1), embedded], dim=1)  # [batch, 1 + embed_dim]

PositionEncoder(20)