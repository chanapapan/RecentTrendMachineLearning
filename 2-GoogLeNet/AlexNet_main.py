#%%
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import torch.nn.functional as F
import numpy as np

#%%
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=preprocess)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])

# Should use a different transformatio for the test set
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=preprocess)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                               shuffle=True, num_workers=2)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                             shuffle=False, num_workers=2)
                                             
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                              shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device', device)


#%%
# Define the models, criterion, optimizer and EPOCH
from def_train_model import train_model

from AlexNetSeq import alex_seq_LRN

alex_seq_LRN = alex_seq_LRN.to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(alex_seq_LRN.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

print(alex_seq_LRN)
#%%
best_model, val_acc_history, loss_history = train_model(alex_seq_LRN, dataloaders, criterion, optimizer, device, 10, 'alex_seq_LRN_lr_0.001_bestsofar')

val_acc_history = np.array(val_acc_history)
np.save('alex_seq_LRN_val_acc_history.npy', val_acc_history)

loss_history = np.array(loss_history)
np.save('alex_seq_LRN_loss_history.npy', loss_history)


#%%
#Test the model
from def_test_model import test_model

# %%
# Load and use the best model for testing the accuracy
alex_seq_LRN.load_state_dict(torch.load('../../weights/googlenet/alex_seq_LRN_lr_0.001_bestsofar.pth'))
test_dataloaders = { 'test': test_dataloader }
test_acc, test_loss = test_model(alex_seq_LRN, test_dataloaders, criterion, device)
