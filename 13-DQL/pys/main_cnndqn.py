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

from replay_buffer import ReplayBuffer
from cnndqn import CNNDQN
from cnnqdn_train_loss_plot import compute_td_loss, plot, train_CNNDQN

# Select GPU or CPU as device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_eps_by_episode(epsilon_start, epsilon_final, epsilon_decay):
    eps_by_episode = lambda episode: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode / epsilon_decay)
    return eps_by_episode

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
eps_by_episode = gen_eps_by_episode(epsilon_start, epsilon_final, epsilon_decay)

env_id = 'SpaceInvaders-v0'
env = gym.make(env_id)

# print(env.unwrapped.get_action_meanings())
# print(env.observation_space.shape)
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
# (210, 160, 3)


import torchvision.transforms as T
from PIL import Image
image_size = 84

transform = T.Compose([T.ToPILImage(),
                       T.Resize((image_size, image_size), interpolation=Image.CUBIC),
                       T.ToTensor()])


model = CNNDQN(3, env.action_space.n, env, device).to(device)
# DQN(
#   (layers): Sequential(
#     (0): Linear(in_features=7056, out_features=128, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=128, out_features=128, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=128, out_features=6, bias=True)
#   )
# )
optimizer = optim.Adam(model.parameters())
replay_buffer = ReplayBuffer(1000)

model, all_rewards, losses = train_CNNDQN(env, model, eps_by_episode, optimizer, replay_buffer, device, transform,image_size, episodes = 10000, batch_size=32, gamma = 0.99)


# import time
# def play_game_CNN(model):
#     done = False
#     obs = env.reset()
#     state = get_state(obs)
#     while(not done):
#         action = model.act(state, epsilon_final)
#         next_obs, reward, done, _ = env.step(action)
#         next_state = get_state2(next_obs)
#         env.render()
#         time.sleep(0.1)
#         state = next_state

# play_game_CNN(model)
# time.sleep(3)
# env.close()
