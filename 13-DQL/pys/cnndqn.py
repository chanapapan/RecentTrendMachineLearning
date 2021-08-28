import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

import gym
import numpy as np

from collections import deque
from tqdm import trange

class CNNDQN(nn.Module):
    def __init__(self, n_channel, n_action, env, device):
        super(CNNDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1= nn.Linear(7*7*64, 512)
        self.fc2= nn.Linear(512, n_action)
        self.env = env
        self.device = device
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def act(self, state, epsilon):
        # get action from policy action and epsilon greedy
        if random.random() > epsilon: # get action from old q-values
            state   = autograd.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True).to(self.device)
            q_value = self.forward(state)
            q_value = q_value.cpu()
            action  = q_value.max(1)[1].item()            
        else: # get random action
            action = random.randrange(self.env.action_space.n)
        return action