import torch
from torch.utils.data import Dataset
import os
from typing import List
import numpy as np
BATCH_SIZE = 64
RESIZE=True
N_ORIG_NEIGHBORS = 200
N_NEAREST_NEIGHBORS = 30
N_NEIGHBORS = 120
N_NEIGHBORS_DATASET=120
class CustomDataset(Dataset):
    def __init__(self, filenames: List[str], N_NEIGHBORS: int, N_ORIG_NEIGHBORS: int):
        self.filenames = filenames
        
        # Load and parse data from .npy files
        self.data = self.load_data()

    
    def load_data(self):
        data_list = []
        for filename in self.filenames:
            
            if os.path.exists(filename):
                data = np.load(filename)
                data_list.append(torch.tensor(data, dtype=torch.float32))
            else:
                raise FileNotFoundError(f"{filename} does not exist.")
        
        # Concatenate all data
        data_tensor = torch.cat(data_list, dim=0)
        return data_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]





