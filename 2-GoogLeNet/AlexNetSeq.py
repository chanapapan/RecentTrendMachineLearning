import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

NUM_CLASSES = 10

alex_seq_LRN = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
    nn.ReLU(inplace=True),
    nn.LocalResponseNorm(size=5,alpha=1e-4, beta=0.75, k=2),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(inplace=True),
    nn.LocalResponseNorm(size=5,alpha=1e-4, beta=0.75, k=2),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),

    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),

    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.AdaptiveAvgPool2d((6, 6)),
    Flatten(),
    nn.Dropout(),

    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),

    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),

    nn.Linear(4096, NUM_CLASSES)
    )

alex_seq = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),

    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),

    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.AdaptiveAvgPool2d((6, 6)),
    Flatten(),
    nn.Dropout(),

    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),

    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    
    nn.Linear(4096, NUM_CLASSES)
    )