
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import utils

log_interval = 100
seed = 1

torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_dir = '../../data/MNIST/dataset' #you can use old downloaded dataset, I use from VGAN
batch_size = 128

train_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, download=True, train=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)


# Reconstruction + KL divergence losses summed over all elements and batch
from models import VAE

model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay = 0.0005)

from train_test import train,test


global plotter, test_recon_img, sample_recon_img
plotter = utils.VisdomLinePlotter(env_name='main')
test_recon_img = utils.VisdomImages(var_name='test_recon_img', env_name='main')
sample_recon_img = utils.VisdomImages(var_name='sample_recon_img', env_name='main')

epochs = 50
for epoch in range(1, epochs + 1):
    train( epoch, model, train_loader, device, optimizer, plotter)
    test( epoch, model, test_loader, device, optimizer, plotter, test_recon_img)

    # RECONSTRUCTED SAMPLE
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        sample_img = sample.view(64, 1, 28, 28)
        sample_recon_img.show_images(sample_img, 'SAMPLE')
        if epoch % 5 == 0:
            print("save image: " + 'results_part1/sample_' + str(epoch) + '.png')
            save_image(sample_img, 'results_part1/sample_' + str(epoch) + '.png')

        

