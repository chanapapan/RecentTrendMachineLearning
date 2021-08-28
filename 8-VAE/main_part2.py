#%%
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import utils
import numpy as np
from PIL import Image
import os

log_interval = 100
seed = 1

torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset

compose = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

ds = torchvision.datasets.ImageFolder(root='ait', transform=compose)

train_dataset, test_dataset = torch.utils.data.random_split(ds, [300, 17])

# train_dataset.dataset.transform = train_preprocess
# val_dataset.dataset.transform = eval_preprocess

batch_size = 100
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#%%

# Reconstruction + KL divergence losses summed over all elements and batch
from models import newVAE, Flatten, UnFlatten

model = newVAE().to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay = 0.0005)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#%%
from train_test import train,test


global plotter, test_recon_img, sample_recon_img
plotter = utils.VisdomLinePlotter(env_name='main')
test_recon_img = utils.VisdomImages(var_name='test_recon_img', env_name='main')
sample_recon_img = utils.VisdomImages(var_name='sample_recon_img', env_name='main')

epochs = 150
for epoch in range(1, epochs + 1):
    train(epoch, model, train_loader, device, optimizer, plotter)
    test(epoch, model, test_loader, device, optimizer, plotter, test_recon_img)

    # RECONSTRUCTED SAMPLE
    with torch.no_grad():
        sample = torch.randn(6, 32).to(device)
        sample = model.decode(sample).cpu()
        sample_recon_img.show_images(sample, 'SAMPLE')
        if epoch % 5 == 0:
            print("save image: " + 'results_part2/sample_' + str(epoch) + '.png')
            save_image(sample, 'results_part2/sample_' + str(epoch) + '.png')

        


# %%
