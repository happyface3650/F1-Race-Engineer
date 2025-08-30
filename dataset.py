import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

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
        #converting all time/flags/boolean values to things that are convertible to floats
        self.df['PitInTime'] = pd.to_timedelta(
        self.df['PitInTime'].replace('NO_PIT', '00:00:00')).dt.total_seconds()
        self.df['PitOutTime'] = pd.to_timedelta(
        self.df['PitOutTime'].replace('NO_PIT', '00:00:00')).dt.total_seconds()
        self.df['TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(self.df['TimeSinceLastWeatherMeasurement']).dt.total_seconds()
        self.df['Rainfall'] = self.df['Rainfall'].astype(int)
        self.df['dnf'] = self.df['dnf'].astype(int)

        self.aux_columns = ['Rainfall', 
                            'status_1','status_2','status_4','status_6','status_12','status_14',
                            'status_16','status_21','status_24','status_26','status_41','status_67',
                            'status_71','status_124','status_126','status_167','status_671','status_712',
                            'status_6712', 'dnf']
        
        
        if self.pit:
            # For each driver, shift PitOutTime backward by 1 lap to match with PitInTime
            self.df = self.df[
                (self.df['PitInTime'] != 0) |
                (self.df['PitOutTime'] != 0)   

            ]
            self.df['NextPitOutTime'] = self.df.groupby('Driver_idx')['PitOutTime'].shift(-1)
            # Compute duration only where PitInTime is not null
            self.df['PitDuration'] = (
                (self.df['NextPitOutTime'] - self.df['PitInTime'])
            )
            self.norm_cols = ['WindSpeed', 'CircuitLength','Number of Laps', 'LapTime_sec', 'GapToLeader', 'GapToAhead' , 'GapToBehind', 'TimeSinceLastWeatherMeasurement',
                                'SpeedFL','SpeedST','TyreLife', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 'LapNumber', 'PitDuration']
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
            self.norm_cols = ['WindSpeed', 'CircuitLength','Number of Laps', 'LapTime_sec', 'GapToLeader', 'GapToAhead' , 'GapToBehind', 'TimeSinceLastWeatherMeasurement',
                                'SpeedFL','SpeedST','TyreLife', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 'LapNumber']
            self.scalers = {col: (self.df[col].mean(), self.df[col].std()) 
                            for col in self.norm_cols if col in self.df.columns}

         # Define feature columns
        
            self.tabular_columns = [col for col in self.df.columns 
                                if col not in ['Name', 'FreshTyre', 'LapStartTime', 'PitInTime', 'PitOutTime', 'RainFall', 
                                                'status_1','status_2','status_4','status_6','status_12','status_14',
                                                'status_16','status_21','status_24','status_26','status_41','status_67',
                                                'status_71','status_124','status_126','status_167','status_671','status_712',
                                                'status_6712', 'Name', 'Driver', 'Team', 'LapStartTime', 'Time_x', 'TimeStamp_y', 'TrackStatus', 
                                                'LapTime_sec', 'Position', 'Unnamed: 0.1', 'Unnamed: 0', 'dnf']]
            self.lap_time = self.df['LapTime_sec'].to_numpy(dtype=np.float32)
            self.position= self.df['Position'].to_numpy(dtype=np.float32)
            self.dnf= self.df['dnf'].to_numpy(dtype=np.float32)

        # Convert to NumPy array (exclude 'Name' and other non-numeric columns)
        self.numerical_data = self.df[self.tabular_columns].to_numpy(dtype=np.float32)
        self.aux_data = self.df[self.aux_columns].to_numpy(dtype=np.float32)

        #target arrays
        

    def __getitem__(self, index):

        row = self.df.iloc[index]
        track_name = row['Name']
        image_embedding = self.track_embeddings[track_name]
        image_tensor = torch.FloatTensor(image_embedding)
        tabular  = torch.FloatTensor(self.numerical_data[index: index+1])
        aux = torch.FloatTensor(self.aux_data[index:index+1])

        if self.pit and self.use_aux:
            target = torch.tensor(self.pitstoptime[index])
            return image_tensor, tabular, aux, target
        elif self.use_aux:
            target1 = torch.tensor(self.lap_time[index])
            target2 = torch.tensor(self.position[index])
            target3 = torch.tensor(self.dnf[index])
            return image_tensor, tabular, aux, target1, target2, target3
        elif self.pit:
            target = torch.tensor(self.pitstoptime[index])
            return image_tensor, tabular, target
        else:
            target1 = torch.tensor(self.lap_time[index])
            target2 = torch.tensor(self.position[index])
            target3 = torch.tensor(self.dnf[index])
            return image_tensor, tabular, target1, target2, target3

# for inference: get image embedding from dict and convert to tensor, get table data from weather predictions, RL agent simulation


