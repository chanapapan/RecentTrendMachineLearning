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

def train(epoch, model, train_loader, device, optimizer, plotter):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_train_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, avg_train_loss))

    plotter.plot('loss', 'train', 'TRAIN and TEST Loss', epoch, avg_train_loss)
    

def test(epoch, model, test_loader, device, optimizer, plotter, test_recon_img):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            # TEST RECONSTRUCTION
            if i == 0:
                n = min(data.size(0), 4)

                # print(data[:n].shape)
                # print(recon_batch[:n].shape)

                comparison = torch.cat([data[:n],recon_batch[:n]])
                test_recon_img.show_images(comparison, 'TEST RECONSTRUCTION')

                if epoch % 5 == 0:
                    save_image(comparison.cpu(),'results_part2/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    plotter.plot('loss', 'test', 'Test Loss', epoch, test_loss)
    
    


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD