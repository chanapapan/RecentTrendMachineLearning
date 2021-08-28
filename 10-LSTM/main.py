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
print(device)

global MAX_LENGTH, MIN_COUNT
MAX_LENGTH = 30
MIN_COUNT = 2 
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

#%%
# Load/Assemble Voc and pairs
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

# Define the models

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
encoder = encoder.to(device)
decoder = decoder.to(device)

#%%

# Define the optimizers and training parameters

from train_eval_loss import trainIters

save_dir = 'content/'
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 2 # 6000
print_every = 10
save_every = 2000
loadFilename = None
corpus_name = "MultiWOZ"

encoder.train()
decoder.train()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

#%%
# Start Training


print("Starting Training!")
lossvalues = trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename, PAD_token, MAX_LENGTH, device, SOS_token, teacher_forcing_ratio)

#%%
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(lossvalues)
plt.show()


#%%
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


# searcher = GreedySearchDecoder(encoder, decoder, device)
searcher = BeamSearchDecoder(encoder, decoder, voc, 10, device)

gram1_bleu_score = []
gram2_bleu_score = []

for i in range(0,len(testpairs),1):
  
    input_sentence = testpairs[i][0]

    reference = testpairs[i][1:]
    templist = []
    for k in range(len(reference)):
        if(reference[k]!=''):
            temp = reference[k].split(' ')
            templist.append(temp)

from preprocess import normalizeString, indexesFromSentence
from train_eval_loss import evaluate, evaluateInput
    input_sentence = normalizeString(input_sentence)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, MAX_LENGTH, device, SOS_token, EOS_token)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    chencherry = SmoothingFunction()

    score1 = sentence_bleu(templist,output_words,weights=(1, 0, 0, 0) ,smoothing_function=chencherry.method1)
    score2 = sentence_bleu(templist,output_words,weights=(0.5, 0.5, 0, 0),smoothing_function=chencherry.method1) 
    gram1_bleu_score.append(score1)
    gram2_bleu_score.append(score2)
    if i % 1000 == 0:
        print(i,sum(gram1_bleu_score)/len(gram1_bleu_score),sum(gram2_bleu_score)/len(gram2_bleu_score))

print("Total Bleu Score for 1 grams on testing pairs: ", sum(gram1_bleu_score)/len(gram1_bleu_score) )  
print("Total Bleu Score for 2 grams on testing pairs: ", sum(gram2_bleu_score)/len(gram2_bleu_score) )

#%%
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
# searcher = GreedySearchDecoder(encoder, decoder, device)
searcher = BeamSearchDecoder(encoder, decoder, voc, device, 10)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc, device, SOS_token, MAX_LENGTH, EOS_token)

# ME : I'm looking for a restaurant.
# Bot: what type of food would you like ? asian oriental ? what area ? is available centre .
# ME : I'm looking for a place to stay.
# Bot: what price range and in what area would you like to stay ? wifi ? wifi town . parking ? is a priced rating .
# ME : I need a lift.
# Error: Encountered unknown word.

#%%

