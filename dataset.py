import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import resize
import torch.nn.functional as F

class ImageTabularDataset(Dataset):
    def __init__(self, image_dir, csv_path=None, target_size=224, use_aux=False, pitstoppred=False):
        
        self.image_dir = image_dir
        self.target_size = target_size
        self.csv_path = csv_path
        self.pit = pitstoppred
        
        if self.csv_path: #with table data
            self.df = pd.read_csv(self.csv_path)
            self.circuit_names = self.df['Name'].values
            self.use_aux = use_aux #flag if aux features e.g. rainfall(bool)/pit times are available
            self.circuit_dims = self._get_circuit_dimensions()
            self.image_cache = {
                circuit: self._load_and_transform(circuit)
                for circuit in self.df['Name'].unique()
            }
            #try with no normalization if it doesnt work well
            
            self._preprocess_data()
        else: #no table data if needed to train CNN on its own
            
            self.circuits = ['Sakhir', 'Jeddah', 'Melbourne', 'Suzuka', 'Shanghai', 
                           'Miami', 'Imola', 'Monaco', 'Montréal', 'Barcelona', 
                           'Spielberg', 'Silverstone', 'Budapest', 'Spa-Francorchamps', 'Zandvoort', 
                           'Monza', 'Baku', 'Marina Bay', 'Austin', 'Mexico City', 
                           'São Paulo', 'Las Vegas', 'Lusail', 'Yas Island']
            self.circuit_dims = self._get_circuit_dimensions()
            self.image_cache = {
                circuit: self._load_and_transform(circuit)
                for circuit in self.circuits
            }
        
    def __len__(self):
        if self.csv_path is not None:
            return len(self.df)
        else:
            return len(self.circuits)

    def _preprocess_data(self):
        #converting all time/flags/boolean values to things that are convertible to floats
        self.df['PitInTime'] = pd.to_timedelta(
        self.df['PitInTime'].replace('NO_PIT', '00:00:00')).dt.total_seconds()
        self.df['PitOutTime'] = pd.to_timedelta(
        self.df['PitOutTime'].replace('NO_PIT', '00:00:00')).dt.total_seconds()
        self.df['TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(self.df['TimeSinceLastWeatherMeasurement']).dt.total_seconds()
        self.df['Rainfall'] = self.df['Rainfall'].astype(int)

        self.aux_columns = ['Rainfall', 
                            'status_1','status_2','status_4','status_6','status_12','status_14',
                            'status_16','status_21','status_24','status_26','status_41','status_67',
                            'status_71','status_124','status_126','status_167','status_671','status_712',
                            'status_6712']
        
        
        if self.pit:
            # For each driver, shift PitOutTime backward by 1 lap to match with PitInTime
            self.df = self.df[
                (self.df['PitInTime'] != 0) |
                (self.df['PitOutTime'] != 0)   

            ]
            print(self.df)
            self.df['NextPitOutTime'] = self.df.groupby('Driver_idx')['PitOutTime'].shift(-1)
            # Compute duration only where PitInTime is not null
            self.df['PitDuration'] = (
                (self.df['NextPitOutTime'] - self.df['PitInTime'])
            )
            print(self.df['PitDuration'])
            self.norm_cols = ['WindSpeed', 'CircuitLength','Number of Laps', 'LapTime_sec', 'GapToLeader', 'GapToAhead' , 'GapToBehind', 'TimeSinceLastWeatherMeasurement',
                                'SpeedFL','SpeedST','TyreLife', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 'LapNumber', 'PitDuration']
            self.scalers = {col: (self.df[col].mean(), self.df[col].std()) 
                            for col in self.norm_cols if col in self.df.columns}
            self.tabular_columns = [col for col in self.df.columns 
                                if col not in ['Name', 'FreshTyre', 'LapStartTime', 'PitInTime', 'PitOutTime', 'RainFall', 
                                                'status_1','status_2','status_4','status_6','status_12','status_14',
                                                'status_16','status_21','status_24','status_26','status_41','status_67',
                                                'status_71','status_124','status_126','status_167','status_671','status_712',
                                                'status_6712', 'Name', 'Driver', 'Team', 'LapStartTime', 'Time_x', 'TimeStamp_y', 'TrackStatus']]
            self.pitstoptime= self.df['PitDuration'].to_numpy(dtype=np.float32)
            
        else:
            self.df = self.df[
            (self.df['PitInTime'] == 'NO_PIT') and  # Never pitted
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
                                                'LapTime_sec', 'Position']]
            self.lap_time = self.df['LapTime_sec'].to_numpy(dtype=np.float32)
            self.position= self.df['Position'].to_numpy(dtype=np.float32)

        # Convert to NumPy array (exclude 'Name' and other non-numeric columns)
        self.numerical_data = self.df[self.tabular_columns].to_numpy(dtype=np.float32)
        self.aux_data = self.df[self.aux_columns].to_numpy(dtype=np.float32)

        #target arrays
        

    def __getitem__(self, index):
        if self.csv_path is not None:
            circuit = self.circuit_names[index]
            image = self.image_cache[circuit]
            tabular  = torch.FloatTensor(self.numerical_data[index: index+1])
            
            if self.use_aux:
                aux = torch.FloatTensor(self.aux_data[index:index+1])
            else: 
                aux = torch.FloatTensor(self.aux_data[index:index+1])
            if self.pit:
                target = torch.tensor(self.pitstoptime)
                return image, tabular, aux, target
            else:
                target1 = torch.tensor(self.lap_time)
                target2 = torch.tensor(self.position)
                return image, tabular, aux, target1, target2
        else:
            circuit = self.circuits[index]
            image = self.image_cache[circuit]

            return image

    
    
    def _get_circuit_dimensions(self):
     
        dims = {}
        
        if self.csv_path is not None:
            circuits = self.df['Name'].unique()
        else:
            circuits = self.circuits
        for circuit in circuits:
                with Image.open(os.path.join(self.image_dir, f"{circuit}.png")) as img:
                    dims[circuit] = img.size
        return dims

    def _load_and_transform(self, circuit_name):
     
        img_path = os.path.join(self.image_dir, f"{circuit_name}.png")
        img = Image.open(img_path).convert('RGB')
        w, h = self.circuit_dims[circuit_name]

        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()

        max_dim = max(w, h)
        pad_w = (max_dim-w)//2
        pad_h = (max_dim-h)//2

        padded = F.pad(
            img_tensor,
            (pad_w, pad_w, pad_h, pad_h),
            mode='constant',
            value=0
        )

        return resize(
            padded.unsqueeze(0),
            size=[self.target_size, self.target_size],
            interpolation=transforms.InterpolationMode.NEAREST
        ).squeeze(0)

train_data = ImageTabularDataset("track_images\\png", "All_Laps_Train.csv", use_aux=True, pitstoppred=True )

