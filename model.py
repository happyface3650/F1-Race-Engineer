import torch
import torch.nn as nn
import math


# ----------------------
# Positional Encoding
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# ----------------------
# StintTransformer
# ----------------------
class StintTransformer(nn.Module):
    def __init__(self, n_drivers, n_teams, n_tyres, n_modes=3,
                 driver_emb_dim=8, team_emb_dim=8, tyre_emb_dim=4, mode_emb_dim=2,
                 n_cont_features=10, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.3):
        super().__init__()
        self.driver_emb = nn.Embedding(n_drivers, driver_emb_dim)
        self.team_emb = nn.Embedding(n_teams, team_emb_dim)
        self.tyre_emb = nn.Embedding(n_tyres, tyre_emb_dim)
        self.mode_emb = nn.Embedding(n_modes, mode_emb_dim)

        input_dim = n_cont_features + driver_emb_dim + team_emb_dim + tyre_emb_dim + mode_emb_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, cont_feats, driver_idx, team_idx, tyre_idx, mode_idx, mask=None):
        batch_size, seq_len, _ = cont_feats.size()
        driver_emb = self.driver_emb(driver_idx).unsqueeze(1).repeat(1, seq_len, 1)
        team_emb = self.team_emb(team_idx).unsqueeze(1).repeat(1, seq_len, 1)
        tyre_emb = self.tyre_emb(tyre_idx)
        mode_emb = self.mode_emb(mode_idx)
        x = torch.cat([cont_feats, driver_emb, team_emb, tyre_emb, mode_emb], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        key_padding_mask = mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        out = self.output_layer(x).squeeze(-1)
        return out
