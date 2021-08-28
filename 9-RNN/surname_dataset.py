#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
#%%

class surname_dataset(Dataset):
    
    def __init__(self, category_lines, transform=None):
        self.n_letters = 57
        self.all_letters = string.ascii_letters + " .,;'"

        all_names = []
        all_categories = list(category_lines.keys())
        all_targets = []
        for i in category_lines.keys():          
                names = category_lines[i]
                category_tensor = torch.full((len(names), 1), all_categories.index(i))
                all_targets.append(category_tensor)
                for each_name in names:
                    this_name_list = []
                    each_name_tensor = self.lineToTensor(each_name)
                    this_name_list.append(each_name_tensor)
                    all_names.append(this_name_list)

        self.data = all_names
        self.target = torch.cat([all_targets[k] for k in range(len(all_categories))], dim=0)
                                                
                                                
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data = self.data[index][0]
        label = self.target[index]
            
        return data, label
    
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor
    
    def letterToIndex(self, letter):
        return self.all_letters.find(letter)
