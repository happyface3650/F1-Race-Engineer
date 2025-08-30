import torch.nn as nn
import torch.nn.functional as F
import torch

class F1LapTimePredictor(nn.Module):
    def __init__(self, num_numeric_features, driver_metadata_dim, team_metadata_dim,
                 track_metadata_dim=2048, 
                 driver_embed_dim=32, team_embed_dim=32, track_embed_proj_dim=256, 
                 transformer_dim=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.driver_embedder = nn.Sequential(
            nn.Linear(driver_metadata_dim, driver_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, driver_embed_dim)
        )

        self.team_embedder = nn.Sequential(
            nn.Linear(team_metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, team_embed_dim)
        )

        self.track_projection = nn.Sequential(
            nn.Linear(track_metadata_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, track_embed_proj_dim)
        )

        self.numeric_projection = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, track_embed_proj_dim)
        )

        total_dim = driver_embed_dim + team_embed_dim + track_embed_proj_dim + 64

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=total_dim, 
                                       nhead=nhead, 
                                       dim_feedforward=transformer_dim, 
                                       dropout=dropout,
                                       batch_first=True),
            num_layers=num_layers
        )

        self.regressor = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.pos_encoder = nn.Parameter(torch.randn(1,1, total_dim))
    def forward(self, numeric_features, track_embedding, driver_metadata, team_metadata):

        driver_embedding = self.driver_embedder(driver_metadata)
        team_embedding = self.team_embedder(team_metadata)
        track_embedding = self.track_projection(track_embedding)
        numeric_features = self.numeric_projection(numeric_features)

        # Concatenate all embeddings
        x = torch.cat((driver_embedding, team_embedding, track_embedding, numeric_features), dim=1)

        x = x.unsqueeze(1)
        x = x + self.pos_encoder

        transformer_out = self.transformer(x)

        prediction = self.regressor(transformer_out.squeeze(1))

        return prediction
    
