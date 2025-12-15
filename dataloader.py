import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def assign_stints(df, car_col='Driver_idx', session_col='Circuit_Name', tyre_life_col='TyreLife'):
    df = df.sort_values([session_col, car_col, 'LapNumber']).reset_index(drop=True)
    df['NewStint'] = df[tyre_life_col] == 0
    df['StintID'] = df.groupby([session_col, car_col])['NewStint'].cumsum()
    return df

# ----------------------
# Dataset
# ----------------------
class StintDataset(Dataset):
    def __init__(self, df, cont_features, target='LapTime_sec',
                 driver_col='Driver_idx', team_col='Team_idx', tyre_col='Compound', mode_col='mode'):
        # Work on a local copy so we don't mutate the caller DataFrame
        df_local = df.copy()
        # Ensure tyre indices start from 0 (original data often uses 1..n)
        if tyre_col in df_local.columns:
            df_local[tyre_col] = df_local[tyre_col].astype(int) - 1
            # Validate indices
            if df_local[tyre_col].min() < 0:
                raise ValueError(f"Found tyre index < 0 after normalization in column '{tyre_col}'")

        self.stints = []
        for (session, car, stint_id), stint_df in df_local.groupby(['Circuit_Name', 'Driver_idx', 'StintID']):
            self.stints.append({
                'cont_feats': torch.tensor(stint_df[cont_features].values, dtype=torch.float),
                'driver_idx': int(stint_df[driver_col].iloc[0]),
                'team_idx': int(stint_df[team_col].iloc[0]),
                'tyre_idx': torch.tensor(stint_df[tyre_col].values, dtype=torch.long) if tyre_col in stint_df.columns else torch.tensor([], dtype=torch.long),
                'mode_idx': torch.tensor(stint_df[mode_col].values, dtype=torch.long) if mode_col in stint_df.columns else torch.tensor([], dtype=torch.long),
                'lap_time': torch.tensor(stint_df[target].values, dtype=torch.float)
            })

    def __len__(self):
        return len(self.stints)

    def __getitem__(self, idx):
        return self.stints[idx]

# ----------------------
# Collate function
# ----------------------
def collate_fn(batch):
    cont_feats_list = [b['cont_feats'] for b in batch]
    driver_idx = torch.tensor([b['driver_idx'] for b in batch], dtype=torch.long)
    team_idx = torch.tensor([b['team_idx'] for b in batch], dtype=torch.long)
    tyre_idx_list = [b['tyre_idx'] for b in batch]
    mode_idx_list = [b['mode_idx'] for b in batch]
    lap_time_list = [b['lap_time'] for b in batch]

    cont_feats_padded = pad_sequence(cont_feats_list, batch_first=True, padding_value=0.0)
    tyre_idx_padded = pad_sequence(tyre_idx_list, batch_first=True, padding_value=0)
    mode_idx_padded = pad_sequence(mode_idx_list, batch_first=True, padding_value=0)
    lap_time_padded = pad_sequence(lap_time_list, batch_first=True, padding_value=0.0)

    seq_lengths = torch.tensor([len(b['lap_time']) for b in batch])
    max_len = cont_feats_padded.size(1)
    mask = torch.arange(max_len).expand(len(batch), max_len) >= seq_lengths.unsqueeze(1)

    return cont_feats_padded, driver_idx, team_idx, tyre_idx_padded, mode_idx_padded, lap_time_padded, mask