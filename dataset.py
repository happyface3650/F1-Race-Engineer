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
    def __init__(self, csv_path, image_dir, target_size=224, use_aux=True):
        self.df = pd.read_csv(csv_path)

        self.image_dir = image_dir

        self.circuit_dims = self._get_circuit_dimensions()
        self.circuit_names = self.df['Name'].values

        self.use_aux = use_aux #flag if aux features e.g. rainfall(bool)/pit times are available
        
        self.target_size = target_size

        self.image_cache = {
            circuit: self._load_and_transform(circuit)
            for circuit in self.df['Name'].unique()
        }
        
        self._preprocess_data()



    def _preprocess_data(self):
        #converting all time/flags/boolean values to things that are convertible to floats
        
        self.df['PitOutTime'] = pd.to_timedelta(self.df['PitOutTime'].replace('NO_PIT', '0 days 00:00:00')).dt.total_seconds()
        self.df['PitInTime'] = pd.to_timedelta(self.df['PitInTime'].replace('NO_PIT', '0 days 00:00:00')).dt.total_seconds()
        self.df['TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(self.df['TimeSinceLastWeatherMeasurement']).dt.total_seconds()
        self.df['Rainfall'] = self.df['Rainfall'].astype(int)

         # Define feature columns
        self.aux_columns = ['PitInTime', 'PitOutTime', 'Rainfall', 
                            'status_1','status_2','status_4','status_6','status_12','status_14',
                            'status_16','status_21','status_24','status_26','status_41','status_67',
                            'status_71','status_124','status_126','status_167','status_671','status_712',
                            'status_6712']
        self.tabular_columns = [col for col in self.df.columns 
                              if col not in ['Name', 'FreshTyre', 'LapStartTime', 'PitInTime', 'PitOutTime', 'RainFall', 
                                             'status_1','status_2','status_4','status_6','status_12','status_14',
                                             'status_16','status_21','status_24','status_26','status_41','status_67',
                                             'status_71','status_124','status_126','status_167','status_671','status_712',
                                             'status_6712', 'Name', 'Driver', 'Team', 'LapStartTime', 'Time_x', 'TimeStamp_y']]

        # Convert to NumPy array (exclude 'Name' and other non-numeric columns)
        self.numerical_data = self.df[self.tabular_columns].to_numpy(dtype=np.float32)
        self.aux_data = self.df[self.aux_columns].to_numpy(dtype=np.float32)

        #target arrays
        self.lap_time = self.df['LapTime_sec'].to_numpy(dtype=np.float32)
        self.position= self.df['Position'].to_numpy(dtype=np.float32)

    def __getitem__(self, index):
        circuit = self.circuit_names[index]
        image = self.image_cache[circuit]
        if self.use_aux:
            aux = torch.FloatTensor(self.aux_data)
        else:
            aux = torch.zeros(24)  # Placeholder for inference
        tabular  = torch.FloatTensor(self.numerical_data)
        target1 = torch.tensor(self.lap_time)
        target2 = torch.tensor(self.position)
        return image, tabular, aux, target1, target2
    
    
    def _get_circuit_dimensions(self):
     
        dims = {}
        for circuit in self.df['Name'].unique():
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
    def get_track_augmentations(target_size=224):
        return transforms.Compose([
            # Safe geometric transforms (no color distortion)
            transforms.RandomAffine(
                degrees=0,  # No rotation (unless tracks are rotation-symmetric)
                translate=(0.05, 0.05),  # Tiny random shifts
            ),
            # Optional: Small zoom (if needed)
            transforms.RandomResizedCrop(
                size=target_size,
                scale=(0.98, 1.02),  # Minimal zoom range
                interpolation=transforms.InterpolationMode.NEAREST,  # Critical for color codes
            ),
            # Non-destructive noise
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
        ])
    
train_data = ImageTabularDataset("All_Laps_Train.csv", "track_images\\png\\train\\tracks", use_aux=True )

