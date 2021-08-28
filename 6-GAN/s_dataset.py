#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
#%%

class s_dataset(Dataset):
    
    def __init__(self, num_sample = 1000, transform=None):
        
        pi = np.pi
        self.data = torch.zeros([num_sample,2])

        for i in range(num_sample):

            theta = torch.FloatTensor(1).uniform_(0, 2*pi)

            r = torch.randn(1)

            x = (10+r) * torch.cos(theta)
            

            if 0.5*pi <= theta and theta <= 1.5*pi:
                y = ((10+r) * torch.sin(theta)) + 10

            else:
                y = ((10+r) * torch.sin(theta)) - 10

            self.data[i,0] = x
            self.data[i,1] = y
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data = self.data[index]
            
        return data
