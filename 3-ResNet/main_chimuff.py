#%%

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt 

# Set device to GPU or CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%%

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder('/root/labs/data/chivsmuff', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16)

# get all 16 images
dataiter = iter(dataloader)
img, label = dataiter.next()

img_0 = img[label==0]
img_1 = img[label==1]
label_0 = label[label==0]
label_1 = label[label==1]

num_data0 = len(img_0)
num_data1 = len(img_1)

def get1folddata():
# random 1 index of each class for validation set
    range_idx0 = np.arange(num_data0)
    range_idx1 = np.arange(num_data1)

    idx_val0 = random.randint(0,num_data0-1)
    idx_val1 = random.randint(0,num_data1-1)

    train_idx0 = range_idx0!=idx_val0
    train_idx1 = range_idx1!=idx_val1

    validation_img = torch.cat([img_0[idx_val0].reshape(1,3,224,224),img_1[idx_val1].reshape(1,3,224,224)],dim=0)
    train_img = torch.cat([img_0[train_idx0].reshape(-1,3,224,224), img_1[train_idx1].reshape(-1,3,224,224)],dim=0)

    validation_label = torch.Tensor([label_0[idx_val0],label_1[idx_val1]]).long()
    train_label = torch.cat([label_0[train_idx0], label_1[train_idx1]],dim=0).long()

    train_ds = TensorDataset(train_img, train_label)
    validation_ds = TensorDataset(validation_img, validation_label)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = 14, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_ds, batch_size = 2, shuffle=True)

    return train_loader, validation_loader

# %%

from ResNet import ResNet
from basic_block import BasicBlock
from SElayer import SELayer
from ResidualSEblock import ResidualSEBasicBlock
from bottleneck_block import BottleneckBlock

def ResSENet18(num_classes = 10):
    return ResNet(ResidualSEBasicBlock, [2, 2, 2, 2], num_classes)

#%%
# Find which parameters get the best validation accuracy
# DO K-FOLD CROSS-VALIDATION

from train_cv import train_model_cv

kfold = 8

all_max_val_acc = []

all_loss_history = []
all_val_acc_history = []

optim_list          = ['SGD','SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam', 'Adam']
lr_list             = [0.001, 0.005, 0.001, 0.005, 0.001, 0.001, 0.005, 0.005] 
weight_decay_list   = [0.0005, 0.0005, 0.0005, 0.0005, 0, 0.0005, 0 , 0.0005]  
momentum_list       = [0, 0, 0.9, 0.9, 0, 0, 0, 0, 0, 0]


for j in range(len(optim_list)):
    
    min_train_loss_each_fold = []
    max_train_acc_each_fold = []
    max_val_acc_each_fold = []

    loss_history_each_fold = []
    vall_acc_history_each_fold = []

    print(f"Model {j}")

    for i in range(kfold):
        
        # GET TRAIN AND VAL SET FOR THIS FOLD
        train_loader, validation_loader = get1folddata()

        dataloaders = {'train': train_loader, 'val': validation_loader}
        
        # Create new model every fold
        resnet = ResSENet18().to(device)
        resnet.load_state_dict(torch.load('../../weights/resnet/resnetSESGDdecay_bestsofar.pth'))

        criterion = nn.CrossEntropyLoss()
        params_to_update = resnet.parameters()

        if optim_list[j] == 'SGD':
            optimizer = optim.SGD(params_to_update, lr = lr_list[j], weight_decay=weight_decay_list[j], momentum = momentum_list[j])
            
        else :
            optimizer = optim.Adam(params_to_update, lr = lr_list[j], weight_decay=weight_decay_list[j])

        
        max_train_acc, max_val_acc, min_train_loss, loss_history, val_acc_history = train_model_cv(model = resnet,
                                                                dataloaders = dataloaders,
                                                                criterion = criterion,
                                                                optimizer = optimizer,
                                                                num_epochs = 20,
                                                                device = device)

        max_val_acc_each_fold.append(max_val_acc)

        loss_history_each_fold.append(loss_history)
        vall_acc_history_each_fold.append(val_acc_history)

        print(f"Fold {i} | max train acc {max_train_acc} | max val acc {max_val_acc} | min train loss {min_train_loss}")

    all_max_val_acc.append(max_val_acc_each_fold)

    all_loss_history.append(loss_history_each_fold)
    all_val_acc_history.append(vall_acc_history_each_fold)

all_max_val_acc = np.array(all_max_val_acc)

all_loss_history = np.array(all_loss_history)
all_val_acc_history = np.array(all_val_acc_history)


# print(all_min_train_loss.shape)
# print(all_max_train_acc.shape)
print(all_max_val_acc.shape)
print(all_loss_history.shape)
print(all_val_acc_history.shape)


#%%

#Get the model with the highest avarage validation accuracy over 8 folds
mean_val_acc = np.mean(all_max_val_acc, axis = 1 )
print(mean_val_acc)

idx_best_model = np.argmax(mean_val_acc)
print(idx_best_model)

#%%

def plot_cv(all_loss_history, all_val_acc_history):
    for i in range(all_loss_history.shape[0]):
        plt.plot(all_loss_history[i,0,:], label=f"Model {i}")
    plt.title("Train Loss")
    plt.ylabel("loss")
    plt.xlabel("EPOCH")
    plt.legend()
    plt.show()
    for i in range(all_val_acc_history.shape[0]):
        plt.plot(all_val_acc_history[i,0,:], label=f"Model {i}")
    plt.title("Validation Accuracy")
    plt.ylabel("acc")
    plt.xlabel("EPOCH")
    plt.legend()
    plt.show()

plot_cv(all_loss_history, all_val_acc_history)

# %%

# train the model with the best parameters we got from cross validation
from train import train_model

train_loader = torch.utils.data.DataLoader(dataset, batch_size = 14, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size = 14, shuffle=True)

dataloaders = {'train': train_loader, 'val': valid_loader}

# Load out trained SENet model
resnet = ResSENet18().to(device)
resnet.load_state_dict(torch.load('../../weights/resnet/resnetSESGDdecay_bestsofar.pth'))

criterion = nn.CrossEntropyLoss()
params_to_update = resnet.parameters()

if optim_list[idx_best_model] == 'SGD':
        optimizer = optim.SGD(params_to_update, lr = lr_list[idx_best_model], weight_decay=weight_decay_list[idx_best_model], momentum = momentum_list[idx_best_model])
else :
        optimizer = optim.Adam(params_to_update, lr = lr_list[idx_best_model], weight_decay=weight_decay_list[idx_best_model])

best_model, val_acc_history, loss_history = train_model(model = resnet,
                                                            dataloaders = dataloaders,
                                                            criterion = criterion,
                                                            optimizer = optimizer,
                                                            num_epochs = 25,
                                                            device = device, 
                                                            weights_name='CHIMUFF')

val_acc_history = np.array(val_acc_history)
np.save('CHIMUFF_val_acc_history.npy', val_acc_history)

loss_history = np.array(loss_history)
np.save('CHIMUFF_loss_history.npy', loss_history)

#%%

from test_chimuff import test_model_chimuff

resnet.load_state_dict(torch.load('../../weights/resnet/CHIMUFF.pth'))

test_dataset = datasets.ImageFolder('test', transform=transform)


test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 8, shuffle=True)

test_dataloaders = { 'test': test_dataloader }
test_acc, test_loss, inputs, preds, labels = test_model_chimuff(resnet, test_dataloaders, criterion, device)
print("Unique predicted labels : ", np.unique(preds.cpu().detach().numpy()))

#%%
classes = ['Chi','Muff', 2,3,4,5,6,7,8,9]

incorrect_im = inputs[labels!=preds].cpu().detach().numpy()

incorrect_im = np.transpose(incorrect_im, (0,2,3,1))
incorrect_im = (1/(2*2.25)) * incorrect_im + 0.5
num_incorrect = incorrect_im.shape[0]
real_label = labels[labels!=preds].cpu().detach().numpy()
preds_label = preds[labels!=preds].cpu().detach().numpy()
real_label = [ classes[i] for i in real_label]
preds_label = [ classes[i] for i in preds_label]

# %%

import numpy as np
import matplotlib.pyplot as plt
rows = 1
cols = num_incorrect
axes = []
fig = plt.figure(figsize=(10,10))
for a in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols, a+1) )
    subplot_title = (f"{real_label[-a].upper()} // {preds_label[-a]}")
    axes[-1].set_title(subplot_title)
    plt.imshow(incorrect_im[-a])
    fig.tight_layout(pad=0)
    fig.show()

# %%
correct_im = inputs[labels==preds].cpu().detach().numpy()

correct_im = np.transpose(correct_im, (0,2,3,1))
correct_im = (1/(2*2.25)) * correct_im + 0.5

num_correct = correct_im.shape[0]
real_label = labels[labels==preds].cpu().detach().numpy()
preds_label = preds[labels==preds].cpu().detach().numpy()
real_label = [ classes[i] for i in real_label]
preds_label = [ classes[i] for i in preds_label]
# %%
import numpy as np
import matplotlib.pyplot as plt
rows = 1
cols = num_correct
axes = []
fig = plt.figure(figsize=(10,10))
for a in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols, a+1) )
    subplot_title = (f"{real_label[-a].upper()} // {preds_label[-a]}")
    axes[-1].set_title(subplot_title)
    plt.imshow(correct_im[-a])
    fig.tight_layout(pad=0)
    fig.show()
    
# %%
