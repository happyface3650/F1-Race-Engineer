import pandas as pd
import torch.nn as nn

df = pd.read_csv('All_Laps_Train.csv')
# Create mappings from categories to integer indices
driver_to_idx = {driver: idx for idx, driver in enumerate(df['Driver'].unique())}
team_to_idx = {team: idx for idx, team in enumerate(df['Team'].unique())}

# Convert to indices
df['Driver_idx'] = df['Driver'].map(driver_to_idx)
df['Team_idx'] = df['Team'].map(team_to_idx)

df = df.drop(['Driver', 'Team'], axis=1)

df.to_csv("All_Laps_Train_embedded")