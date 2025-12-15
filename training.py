import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from dataloader import assign_stints, StintDataset, collate_fn
from model import StintTransformer
from sklearn.metrics import mean_absolute_error
import joblib
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
# 1. Load and preprocess data
df = pd.read_csv('labeled_ALL.csv')
df = assign_stints(df)

# 2. Continuous features
cont_features = ['LapNumber', 'TyreLife', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall',
                 'TrackTemp', 'WindDirection', 'WindSpeed', 'Circuit_CircuitLength',
                 'Circuit_Number_of_Laps', 'Circuit_NumberOfTurns', 'Circuit_AverageAngleAbs',
                 'Circuit_AverageAngle', 'GapToLeader', 'GapToAhead',
                 'GapToBehind', 'status_1', 'status_12', 'status_124',
                 'status_21', 'status_24', 'status_4',
                 'status_41',  # Add this line
                 "TimeSinceLastWeatherMeasurement", "dnf"]

# 3. Create dataset and dataloader
dataset = StintDataset(df, cont_features)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 4. Model
n_drivers = df['Driver_idx'].nunique()
n_teams = df['Team_idx'].nunique()
n_tyres = int((df['Compound'] - 1).max() + 1)  # compute after converting to 0-based indices
n_modes = df['mode'].nunique()
n_cont_features = len(cont_features)

model = StintTransformer(n_drivers=n_drivers, n_teams=n_teams, n_tyres=n_tyres,
                         n_modes=n_modes, n_cont_features=n_cont_features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_and_save(model, dataloader, optimizer, criterion, device, num_epochs=19, save_path='stint_transformer_model.pth', use_plateau: bool = True, plateau_kwargs: dict = None):
    """Train provided model and save weights to `save_path`.

    Returns the trained model.
    """
    model.to(device)
    print(f"Starting training for {num_epochs} epochs")
    scheduler = None
    if use_plateau:
        plateau_kwargs = plateau_kwargs or {}
        # sensible defaults: halve LR on plateau, wait 3 epochs
        defaults = dict(mode='min', factor=0.5, patience=3)
        merged = {**defaults, **plateau_kwargs}
        try:
            scheduler = ReduceLROnPlateau(optimizer, **merged)
            print(f"ReduceLROnPlateau scheduler enabled with params: {merged}")
        except TypeError as e:
            # Some torch versions may not support newer kwargs; fall back to a minimal scheduler
            print(f"Warning: ReduceLROnPlateau init failed ({e}); scheduler disabled.")
            scheduler = None
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for cont_feats, driver_idx, team_idx, tyre_idx, mode_idx, lap_time, mask in dataloader:
            # Move to device
            cont_feats = cont_feats.to(device)
            driver_idx = driver_idx.to(device)
            team_idx = team_idx.to(device)
            tyre_idx = tyre_idx.to(device)
            mode_idx = mode_idx.to(device)
            lap_time = lap_time.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(cont_feats, driver_idx, team_idx, tyre_idx, mode_idx, mask)

            # Mask out padded values
            loss = criterion(outputs[~mask], lap_time[~mask])

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_metric = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_metric:.4f}")
        # Step scheduler on the monitored metric (training loss here)
        if scheduler is not None:
            scheduler.step(epoch_metric)
            # print current LR(s)
            lrs = {i: g['lr'] for i, g in enumerate(optimizer.param_groups)}
            print(f"Learning rates: {lrs}")

    torch.save(model.state_dict(), save_path)
    model.eval()
    print(f"Model saved to {save_path}")
    return model


def predict_and_evaluate(model_or_path, device, predict_csv: str, out_csv: str = 'PREDICTED.csv', label_threshold_sec: float = 1.0, target_mean: float = None, target_std: float = None, compute_pace_label: bool = False):
    """Run model predictions on `predict_csv`, compute MAE and save predictions.

    - model_or_path: model object or path to state_dict. If path is provided, loads into module `model`.
    - Returns: (mae, out_df)
    """
    df_pred = pd.read_csv(predict_csv)
    df_pred = assign_stints(df_pred)

    pred_dataset = StintDataset(df_pred, cont_features)
    pred_loader = DataLoader(pred_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    preds_list = []
    actuals_list = []
    indices_list = []
    stint_counter = 0

    # load model if a path provided
    if isinstance(model_or_path, str):
        model.load_state_dict(torch.load(model_or_path, map_location=device))
        model.to(device)
        m = model
    else:
        m = model_or_path

    with torch.no_grad():
        for cont_feats, driver_idx, team_idx, tyre_idx, mode_idx, lap_time, mask in pred_loader:
            cont_feats = cont_feats.to(device)
            driver_idx = driver_idx.to(device)
            team_idx = team_idx.to(device)
            tyre_idx = tyre_idx.to(device)
            mode_idx = mode_idx.to(device)
            mask = mask.to(device)

            out = m(cont_feats, driver_idx, team_idx, tyre_idx, mode_idx, mask)
            out = out.cpu()

            for i in range(out.size(0)):
                valid_len = (~mask[i]).sum().item()
                preds_seq = out[i, :valid_len].tolist()
                actuals_seq = lap_time[i, :valid_len].tolist()

                stint_info = pred_dataset.stints[stint_counter]
                inds = stint_info.get('indices', list(range(valid_len)))[:valid_len]

                preds_list.extend(preds_seq)
                actuals_list.extend(actuals_seq)
                indices_list.extend(inds)
                stint_counter += 1

    res_df = pd.DataFrame({'orig_index': indices_list, 'pred': preds_list, 'actual': actuals_list})
    out_df = df_pred.reset_index().rename(columns={'index': 'orig_index'})
    out_df = out_df.merge(res_df, on='orig_index', how='left')

    # Prefer to compute MAE in seconds if possible (either via passed mean/std or saved scalers)
    mae_in_seconds = None
    if (target_mean is not None) and (target_std is not None):
        out_df['pred_sec'] = out_df['pred'] * target_std + target_mean
        out_df['actual_sec'] = out_df['actual'] * target_std + target_mean
        mae_in_seconds = mean_absolute_error(out_df['actual_sec'].dropna(), out_df['pred_sec'].dropna())
        print(f'MAE on {predict_csv}: {mae_in_seconds:.4f} seconds (rescaled using provided mean/std)')
    else:
        # Try to auto-load scalers saved during normalization for this CSV
        scaler_path = os.path.splitext(predict_csv)[0] + '_scalers.joblib'
        loaded_scalers = None
        if os.path.exists(scaler_path):
            try:
                loaded_scalers = joblib.load(scaler_path)
                if 'lap_mean' in loaded_scalers and 'lap_std' in loaded_scalers:
                    tm = loaded_scalers['lap_mean']
                    ts = loaded_scalers['lap_std']
                    out_df['pred_sec'] = out_df['pred'] * ts + tm
                    out_df['actual_sec'] = out_df['actual'] * ts + tm
                    mae_in_seconds = mean_absolute_error(out_df['actual_sec'].dropna(), out_df['pred_sec'].dropna())
                    print(f'MAE on {predict_csv}: {mae_in_seconds:.4f} seconds (rescaled using {scaler_path})')
                else:
                    mae = mean_absolute_error(out_df['actual'].dropna(), out_df['pred'].dropna())
                    print(f'MAE on {predict_csv}: {mae:.4f} (same units as training target)')
            except Exception as e:
                print(f'Failed to load scalers from {scaler_path}: {e}')
                mae = mean_absolute_error(out_df['actual'].dropna(), out_df['pred'].dropna())
                print(f'MAE on {predict_csv}: {mae:.4f} (same units as training target)')
        else:
            mae = mean_absolute_error(out_df['actual'].dropna(), out_df['pred'].dropna())
            print(f'MAE on {predict_csv}: {mae:.4f} (same units as training target)')

    # Compute residuals relative to seconds if available, otherwise in training units
    if 'pred_sec' in out_df.columns and 'actual_sec' in out_df.columns:
        out_df['residual_sec'] = out_df['actual_sec'] - out_df['pred_sec']
        if compute_pace_label:
            def label_row_seconds(r):
                if pd.isna(r):
                    return None
                if r <= -label_threshold_sec:
                    return 'pushing'
                elif r >= label_threshold_sec:
                    return 'conserving'
                else:
                    return 'doing_nothing'
            out_df['pace_label'] = out_df['residual_sec'].apply(label_row_seconds)
    else:
        out_df['residual'] = out_df['actual'] - out_df['pred']
        if compute_pace_label:
            def label_row_unit(r):
                if pd.isna(r):
                    return None
                if r <= -label_threshold_sec:
                    return 'pushing'
                elif r >= label_threshold_sec:
                    return 'conserving'
                else:
                    return 'doing_nothing'
            out_df['pace_label'] = out_df['residual'].apply(label_row_unit)
    out_path = out_csv or predict_csv.replace('.csv', '_preds.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Predictions written to {out_path}.')
    if 'pace_label' in out_df.columns:
        print('Label counts:')
        print(out_df['pace_label'].value_counts(dropna=True))
    else:
        print('No `pace_label` column (set compute_pace_label=True to enable labeling).')

    return mae, out_df

trained = train_and_save(model, dataloader, optimizer, criterion, device, num_epochs=25, save_path='stint_transformer_model.pth')
mae, out_df = predict_and_evaluate('stint_transformer_model.pth', device, 'labeled_TEST.csv')
