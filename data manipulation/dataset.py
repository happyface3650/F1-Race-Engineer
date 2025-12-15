import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class TabularDataset(Dataset):
    def __init__(self, csv_path, mae=0):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        self.mae = mae
        self._preprocess_data()
        
    def __len__(self):
        return len(self.df)

    def _preprocess_data(self):
         # Define feature columns
        self.tabular_columns = [col for col in self.df.columns 
                                if col not in ['Name', 'FreshTyre', 'LapStartTime', 'PitInTime', 'PitOutTime', 'RainFall', 
                                                'status_1','status_2','status_4','status_6','status_12','status_14',
                                                'status_16','status_21','status_24','status_26','status_41','status_67',
                                                'status_71','status_124','status_126','status_167','status_671','status_712',
                                                'status_6712', 'Name', 'Driver', 'Team', 'LapStartTime', 'Time_x', 'TimeStamp_y', 'TrackStatus', 
                                                'LapTime_sec', 'Position', 'Unnamed: 0.1', 'Unnamed: 0', 'delta', 'predicted_base_pace', 'TyreLife', 'LapNumber']]
        self.lap_time = self.df['LapTime_sec'].to_numpy(dtype=np.float32)

        # Convert to NumPy array (exclude 'Name' and other non-numeric columns)
        self.numerical_data = self.df[self.tabular_columns].to_numpy(dtype=np.float32)

    def __getitem__(self, index):

        row = self.df.iloc[index]
        tabular  = torch.FloatTensor(self.numerical_data[index: index+1])
        target = torch.tensor(self.lap_time[index])
        return tabular, target


# for inference: get image embedding from dict and convert to tensor, get table data from weather predictions, RL agent simulation


