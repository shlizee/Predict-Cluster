from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn

from torch import optim
import torch.nn.functional as F

import numpy as np
import math
from torch.utils.data import random_split
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network module only set encoder to be bidirection
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.num_layers = num_layers

    def forward(self, input_tensor, seq_len):
        
        encoder_hidden = torch.Tensor().to(device)
        
        for it in range(max(seq_len)):
          if it == 0:
            enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it+1, :])
          else:
            enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it+1, :], hidden_tmp)
          encoder_hidden = torch.cat((encoder_hidden, enout_tmp),1)

        hidden = torch.empty((1, len(seq_len), encoder_hidden.shape[-1])).to(device)
        count = 0
        for ith_len in seq_len:
            hidden[0, count, :] = encoder_hidden[count, ith_len - 1, :]
            count += 1
        
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(output_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)

        output = self.out(output)
        return output, hidden

class autoencoder(nn.Module):
  def __init__(self, input_size, middle_size):
      super(autoencoder, self).__init__()
      self.encoder = nn.Sequential(
          nn.Linear(input_size, 1024),
          nn.Tanh(),
          nn.Linear(1024, 512), 
          nn.Tanh(),
          nn.Linear(512, middle_size), 
          nn.Tanh()
          )
      
      self.decoder = nn.Sequential(
          nn.Linear(middle_size, 512),
          nn.Tanh(),
          nn.Linear(512, 1024), 
          nn.Tanh(),
          nn.Linear(1024, input_size),
          )

  def forward(self, x):
      middle_x = self.encoder(x)
      x = self.decoder(middle_x)
      return x, middle_x
      
class seq2seq(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, output_size, batch_size,
                 en_num_layers=3, de_num_layers=1,
                 fix_state=False, fix_weight=False, teacher_force=False):
        super(seq2seq, self).__init__()
        self.batch_size = batch_size
        self.en_num_layers = en_num_layers
        self.encoder = EncoderRNN(en_input_size, en_hidden_size, en_num_layers).to(device)
        self.decoder = DecoderRNN(output_size, en_hidden_size * 2, de_num_layers).to(device)
        self.fix_state = fix_state
        self.fix_weight = fix_weight
        self.device = device
        if self.fix_weight:
            with torch.no_grad():
                # decoder fix weight
                self.decoder.gru.requires_grad = False
                # self.decoder.out.requires_grad = False
                
        self.en_input_size = en_input_size
        self.teacher_force = teacher_force

    def forward(self, input_tensor, seq_len):
        self.batch_size = len(seq_len)
        
        encoder_hidden = self.encoder(
            input_tensor, seq_len)

        decoder_output = torch.Tensor().to(self.device)
        
        # decoder part
        if self.teacher_force:
          de_input = torch.zeros([self.batch_size, 1, self.en_input_size], dtype=torch.float).to(device)
          de_input = torch.cat((de_input, input_tensor[:,1:,:]), dim = 1)
        else:
          de_input = torch.zeros(input_tensor.size(), dtype=torch.float).to(device)

        if self.fix_state:
                
            de_input = input_tensor[:,0:1, :]

            for it in range(max(seq_len)):
                deout_tmp, _ = self.decoder(de_input, encoder_hidden)
                deout_tmp = deout_tmp + de_input
                de_input = deout_tmp
                decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)
        else:
            hidden = encoder_hidden
            for it in range(max(seq_len)):
                deout_tmp, hidden = self.decoder(
                    de_input[:, it:it+1, :], hidden)

                decoder_output = torch.cat((decoder_output, deout_tmp), dim=1)
        return encoder_hidden, decoder_output