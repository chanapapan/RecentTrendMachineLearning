#%%
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import torch.nn.functional as F
import numpy as np

# Set device to GPU or CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%%
# Allow augmentation transform for training set, no augementation for val/test set

train_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

eval_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Download CIFAR-10 and split into training, validation, and test sets.
# The copy of the training dataset after the split allows us to keep
# the same training/validation split of the original training set but
# apply different transforms to the training set and validation set.

full_train_dataset = torchvision.datasets.CIFAR10(root='/root/labs/data', train=True,
                                                  download=True)

train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [40000, 10000])

train_dataset.dataset = copy(full_train_dataset)
train_dataset.dataset.transform = train_preprocess
val_dataset.dataset.transform = eval_preprocess

test_dataset = torchvision.datasets.CIFAR10(root='/root/labs/data', train=False,
                                            download=True, transform=eval_preprocess)

# DataLoaders for the three datasets

BATCH_SIZE = 16
NUM_WORKERS = 2

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

#%%
from ResNet import ResNet
from basic_block import BasicBlock
from SElayer import SELayer
from ResidualSEblock import ResidualSEBasicBlock
from bottleneck_block import BottleneckBlock

def ResNet18(num_classes = 10):
    '''
    First conv layer: 1
    4 residual blocks with two sets of two convolutions each: 2*2 + 2*2 + 2*2 + 2*2 = 16 conv layers
    last FC layer: 1
    Total layers: 1+16+1 = 18
    '''
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResSENet18(num_classes = 10):
    return ResNet(ResidualSEBasicBlock, [2, 2, 2, 2], num_classes)

#%%

from train import train_model

# Create ResSENet18 Model
resnet = ResSENet18().to(device)

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
params_to_update = resnet.parameters()

# Now we'll use Adam optimization
# optimizer = optim.Adam(params_to_update, lr=0.01)
optimizer = optim.SGD(params_to_update, lr=0.01, weight_decay=0.0005)


best_model, val_acc_history, loss_history = train_model(model = resnet,
                                                            dataloaders = dataloaders,
                                                            criterion = criterion,
                                                            optimizer = optimizer,
                                                            num_epochs = 25,
                                                            device = device, 
                                                            weights_name='resnetSEADAMdecay_bestsofar')
#%%

val_acc_history = np.array(val_acc_history)
np.save('resnetSEADMdecay_val_acc_history.npy', val_acc_history)

loss_history = np.array(loss_history)
np.save('resnetSEADAMdecay_loss_history.npy', loss_history)


# %%
from test import test_model

resnet.load_state_dict(torch.load('../../weights/resnet/resnetSEADAMdecay_bestsofar.pth'))

test_dataloaders = { 'test': test_dataloader }
test_acc, test_loss = test_model(resnet, test_dataloaders, criterion, device)

# %%
