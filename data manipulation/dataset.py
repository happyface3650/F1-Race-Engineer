import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class ImageTabularDataset(Dataset):
    def __init__(self, track_embeddings_dict, csv_path, use_aux=False, pitstoppred=False):
        
        self.track_embeddings = track_embeddings_dict
        self.csv_path = csv_path
        self.pit = pitstoppred
        self.df = pd.read_csv(self.csv_path)
        self.use_aux = use_aux

        self._preprocess_data()
        
    def __len__(self):
        return len(self.df)

    def _preprocess_data(self):

        self.aux_columns = ['Rainfall', 
                            'status_1','status_2','status_4','status_6','status_12','status_14',
                            'status_16','status_21','status_24','status_26','status_41','status_67',
                            'status_71','status_124','status_126','status_167','status_671','status_712',
                            'status_6712', 'dnf', 'SpeedFL', 'SpeedST']
        
        
        if self.pit:
            
            self.norm_cols = ['WindSpeed', 'CircuitLength','Number of Laps', 'LapTime_sec', 'GapToLeader', 'GapToAhead' , 'GapToBehind', 'TimeSinceLastWeatherMeasurement', 'TyreLife', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 'LapNumber', 'PitDuration']
            self.scalers = {col: (self.df[col].mean(), self.df[col].std()) 
                            for col in self.norm_cols if col in self.df.columns}
            self.tabular_columns = [col for col in self.df.columns 
                                if col not in ['Name', 'FreshTyre', 'LapStartTime', 'PitInTime', 'PitOutTime', 'RainFall', 
                                                'status_1','status_2','status_4','status_6','status_12','status_14',
                                                'status_16','status_21','status_24','status_26','status_41','status_67',
                                                'status_71','status_124','status_126','status_167','status_671','status_712',
                                                'status_6712', 'Name', 'Driver', 'Team', 'LapStartTime', 'Time_x', 'TimeStamp_y', 'TrackStatus', 'Unnamed: 0.1', 'Unnamed: 0', 'NextPitOutTime'
                                                , 'dnf']]
            self.pitstoptime= self.df['PitDuration'].to_numpy(dtype=np.float32)

            
        else:
            self.df = self.df[
            (self.df['PitInTime'] == 'NO_PIT') &  # Never pitted
            (self.df['PitOutTime'] == 'NO_PIT')   # Never pitted

        ]
        #normalization
            self.norm_cols = ['WindSpeed', 'CircuitLength','Number of Laps', 'LapTime_sec', 'GapToLeader', 'GapToAhead' , 'GapToBehind', 'TimeSinceLastWeatherMeasurement', 'TyreLife', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 'LapNumber']
            self.scalers = {col: (self.df[col].mean(), self.df[col].std()) 
                            for col in self.norm_cols if col in self.df.columns}

         # Define feature columns
        
            self.tabular_columns = [col for col in self.df.columns 
                                if col not in ['Name', 'FreshTyre', 'LapStartTime', 'PitInTime', 'PitOutTime', 'RainFall', 
                                                'status_1','status_2','status_4','status_6','status_12','status_14',
                                                'status_16','status_21','status_24','status_26','status_41','status_67',
                                                'status_71','status_124','status_126','status_167','status_671','status_712',
                                                'status_6712', 'Name', 'Driver', 'Team', 'LapStartTime', 'Time_x', 'TimeStamp_y', 'TrackStatus', 
                                                'LapTime_sec', 'Position', 'Unnamed: 0.1', 'Unnamed: 0']]
            self.lap_time = self.df['LapTime_sec'].to_numpy(dtype=np.float32)

        # Convert to NumPy array (exclude 'Name' and other non-numeric columns)
        self.numerical_data = self.df[self.tabular_columns].to_numpy(dtype=np.float32)
        self.aux_data = self.df[self.aux_columns].to_numpy(dtype=np.float32)
        
        # Apply normalization to numerical data
        self._normalize_data()

        #target arrays
        
    def _normalize_data(self):
        """Apply Z-score normalization using sklearn"""
        scaler = StandardScaler()
        self.numerical_data = scaler.fit_transform(self.numerical_data)
        self.scaler = scaler  # Save for later use (inference)

    def __getitem__(self, index):

        row = self.df.iloc[index]
        track_name = row['Name']
        image_embedding = self.track_embeddings[track_name]
        self.df.drop(['Name'], axis=1)
        image_tensor = torch.FloatTensor(image_embedding)
        tabular  = torch.FloatTensor(self.numerical_data[index: index+1])
        aux = torch.FloatTensor(self.aux_data[index:index+1])

        if self.use_aux:
            target = torch.tensor(self.lap_time[index])
            return image_tensor, tabular, aux, target
        else:
            target = torch.tensor(self.lap_time[index])
            return image_tensor, tabular, target


# for inference: get image embedding from dict and convert to tensor, get table data from weather predictions, RL agent simulation


