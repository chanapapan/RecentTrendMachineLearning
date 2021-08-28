#%%
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pickle
from torch.autograd import Variable
from get_free_gpu import get_free_gpu

import train_eval_loss
from model import EncoderRNN, Attn, LuongAttnDecoderRNN, GreedySearchDecoder, Sentence, beam_decode, BeamSearchDecoder
from lstm_cell import NaiveCustomLSTM
import preprocess
from preprocess import Voc

CUDA = torch.cuda.is_available()
device = get_free_gpu()

# global MAX_LENGTH, MIN_COUNT, PAD_token, SOS_token, EOS_token
MAX_LENGTH = 30
MIN_COUNT = 2 
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

#%%
from preprocess import loadPrepareData
datafile = '../../data/multiwoz/multiwoz_all.txt'
voc, pairs = loadPrepareData(datafile, MAX_LENGTH)

# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

#%%
# Trim vocabulary and pairs
from preprocess import trimRareWords
pairs_ = trimRareWords(voc, pairs, MIN_COUNT)

testpairs = pairs_[5000:]
pairs  = pairs_[:5000]

print(len(pairs))
print(len(testpairs))

#%%
from preprocess import batch2TrainData
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)], PAD_token)
input_variable, lengths, target_variable, mask, max_target_len = batches

#%%
pair_batch = pairs[:5]
print(pair_batch)
pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
print(pair_batch)
print(target_variable)
print(mask)
print(max_target_len)

#%%
model_name = 'cb_model'
attn_model = 'dot'

hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 4
dropout = 0.5
batch_size = 256 
loadFilename = None

embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

saved_dict = torch.load('content/cb_model/MultiWOZ/en2-de4_beam/6000_checkpoint.tar')

embedding.load_state_dict(saved_dict['embedding'])
encoder.load_state_dict(saved_dict['en'])
decoder.load_state_dict(saved_dict['de'])

encoder = encoder.to(device)
decoder = decoder.to(device)

#%%
learning_rate = 0.0001
decoder_learning_ratio = 5.0

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

encoder_optimizer.load_state_dict(saved_dict['en_opt'])
decoder_optimizer.load_state_dict(saved_dict['de_opt'])

#%%

from train_eval_loss import evaluateInput
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder, device)
# searcher = BeamSearchDecoder(encoder, decoder, voc, device, 10)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc, device, SOS_token, MAX_LENGTH, EOS_token)
# %%
