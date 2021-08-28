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
from dqn import DQN
from train_loss_plot import compute_td_loss, plot, train

# Select GPU or CPU as device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_eps_by_episode(epsilon_start, epsilon_final, epsilon_decay):
    eps_by_episode = lambda episode: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode / epsilon_decay)
    return eps_by_episode

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
eps_by_episode = gen_eps_by_episode(epsilon_start, epsilon_final, epsilon_decay)

env_id = "CartPole-v0"
env = gym.make(env_id)

model = DQN(env.observation_space.shape[0], env.action_space.n, env, device).to(device)
    
optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayBuffer(1000)

model,all_rewards, losses = train(env, model, eps_by_episode, optimizer, replay_buffer, device,episodes = 10000, batch_size=32, gamma = 0.99)


# import time
# def play_game(model):
#     done = False
#     state = env.reset()
#     while(not done):
#         action = model.act(state, epsilon_final)
#         next_state, reward, done, _ = env.step(action)
#         env.render()
#         time.sleep(0.03)
#         state = next_state

# play_game(model)
# time.sleep(3)
# env.close()
