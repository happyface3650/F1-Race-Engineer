import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np

from torchvision import transforms

class ImageTabularDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform, use_aux=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.circuit_names = self.df['Name'].values #circuit name as nparrauy
        self.transform = transform
        self.use_aux = use_aux #flag if aux features e.g. rainfall(bool)/pit times are available
        
        self.image_cache = {
            circuit: self.transform(Image.open(os.path.join(image_dir, f"{circuit}.png")).convert('RGB'))
            for circuit in self.df['Name'].unique()
        }
        
        self._preprocess_data()



    def _preprocess_data(self):
        #converting all time/flags/boolean values to things that are convertible to floats
        self.df["Time_x"] = pd.to_timedelta(self.df["Time_x"]).dt.total_seconds()
        self.df["Timestamp_y"] = pd.to_timedelta(self.df["Timestamp_y"]).dt.total_seconds()
        self.df['PitOutTime'] = pd.to_timedelta(self.df['PitOutTime'].replace('NO_PIT', '0 days 00:00:00')).dt.total_seconds()
        self.df['PitInTime'] = pd.to_timedelta(self.df['PitInTime'].replace('NO_PIT', '0 days 00:00:00')).dt.total_seconds()
        self.df['RainFall'] = self.df['RainFall'].astype(int)

         # Define feature columns
        self.aux_columns = ['PitInTime', 'PitOutTime', 'RainFall']
        self.tabular_columns = [col for col in self.df.columns 
                              if col not in ['Name', 'FreshTyre', 'LapStartTime', 'PitInTime', 'PitOutTime', 'RainFall']]

        # Convert to NumPy array (exclude 'Name' and other non-numeric columns)
        self.numerical_data = self.df[self.tabular_columns].to_numpy(dtype=np.float32)
        self.aux_data = self.df[self.aux_columns].to_numpy(dtype=np.float32)

        #target arrays
        self.lap_time = self.df[['LapTime']].to_numpy(dtype=np.float32)
        self.position= self.df[['Position']].to_numpy(dtype=np.float32)



def __getitem__(self, index):
        circuit = self.circuit_names[index]
        image = self.image_cache[circuit]
        if self.use_aux:
            aux = torch.FloatTensor(self.aux_data)
        else:
            aux = torch.zeros(3)  # Placeholder for inference
        tabular  = torch.FloatTensor(self.numerical_data)
        target1 = torch.tensor(self.lap_time)
        target2 = torch.tensor(self.position)
        return image, tabular, aux, target1, target2



transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


train_data = ImageTabularDataset("All_Laps_Train.csv", "track_images\\png\\train\\tracks", transform, True )

