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
import urllib

#%%
# Preprocess inputs to 3x32x32 with CIFAR-specific normalization parameters

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Download CIFAR-10 and set up train, validation, and test datasets with new preprocess object
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=preprocess)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])

# Should use a different transformatio for the test set
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_preprocess)

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
from GoogleNet import GoogLeNet, Inception

GoogLeNet = GoogLeNet().to(device)

#===================================================
# PRETRAINED MODEL
# googlenet_pre = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True, aux_logits = True)
# googlenet_pre.aux1.fc2 = nn.Linear(1024,10)
# print(googlenet_pre.aux1.fc2)
# googlenet_pre.aux2.fc2 = nn.Linear(1024,10)
# print(googlenet_pre.aux2.fc2)
# googlenet_pre.fc = nn.Linear(1024,10)
# print(googlenet_pre.fc)
# googlenet_pre = googlenet_pre.to(device)
#====================================================

criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(googlenet.parameters(), lr=0.01)
optimizer = optim.SGD(GoogLeNet.parameters(), lr=0.01)
# optimizer = optim.SGD(googlenet_pre.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

#%%
best_model, val_acc_history, loss_history = train_model(model = GoogLeNet,
                                                        dataloaders = dataloaders,
                                                        criterion = criterion,
                                                        optimizer = optimizer,
                                                        device = device,
                                                        num_epochs = 10,
                                                        weights_name = 'GoogLeNet_lr_0.01_bestsofar',
                                                        is_inception=True)

val_acc_history = np.array(val_acc_history)
np.save('GoogLeNet_val_acc_history.npy', val_acc_history)

loss_history = np.array(loss_history)
np.save('GoogLeNet_loss_history.npy', loss_history)

#%%
from def_test_model import test_model

GoogLeNet.load_state_dict(torch.load('../../weights/googlenet/GoogLeNet_lr_0.01_bestsofar.pth'))
test_dataloaders = { 'test': test_dataloader }
test_acc, test_loss = test_model(GoogLeNet, test_dataloaders, criterion, device)

# %%
