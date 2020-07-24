# load file
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from PCNet import *
from utilitiesPC import *
from torch import optim
import torch.nn.functional as F
from data_loaderPC import *
import h5py
import numpy as np

import time
import math
from torch.utils.data import random_split
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_iter(input_tensor, seq_len,  model, optimizer, criterion_seq):
    optimizer.zero_grad()
    
    en_hi, de_out = model(input_tensor, seq_len)


    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    mask = torch.sum(mask, 1)

    total_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
    total_loss = torch.mean(torch.sum(total_loss, 1) / mask)
    
    total_loss.backward()
    clip_grad_norm_(model.parameters(), 25 , norm_type=2)
    
    optimizer.step()
    return total_loss, en_hi


def eval_iter(input_tensor, seq_len, model, criterion_seq):
    
    en_hi, de_out = model(input_tensor, seq_len)

    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    mask = torch.sum(mask, 1)

    total_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
    total_loss = torch.mean(torch.sum(total_loss, 1) / mask)

    return total_loss, en_hi


def evaluation(validation_loader, model, criterion_seq):
    total_loss = 0

    for ind, (eval_data, seq_len, label) in enumerate(validation_loader):
        input_tensor = eval_data.to(device)
        loss, hid = eval_iter(input_tensor, seq_len, model,
                                        criterion_seq)
        total_loss += loss.item()

    ave_loss = total_loss / (ind + 1)
    return ave_loss

def test_extract_hidden(model, data_train, data_eval):
    label_list_train = []
    label_list_eval = []

    for ith, (ith_data, seq_len, label) in enumerate(data_train):
        input_tensor = ith_data.to(device)
        
        en_hi, de_out = model(input_tensor, seq_len)


        if ith == 0:
            label_train = label
            hidden_array_train = en_hi[0, :, :].detach().cpu().numpy()

        else:
            label_train = np.hstack((label_train, label))
            hidden_array_train = np.vstack((hidden_array_train, en_hi[0, :, :].detach().cpu().numpy()))

    for ith, (ith_data, seq_len, label) in enumerate(data_eval):

        input_tensor = ith_data.to(device)

        en_hi, de_out = model(input_tensor, seq_len)

        if ith == 0:
            hidden_array_eval = en_hi[0, :, :].detach().cpu().numpy()
            label_eval = label
        else:
            label_eval =  np.hstack((label_eval, label))
            hidden_array_eval = np.vstack((hidden_array_eval, en_hi[0, :, :].detach().cpu().numpy()))

    return hidden_array_train, hidden_array_eval, label_train, label_eval


def train_autoencoder(hidden_train, hidden_eval, label_train,
                      label_eval, middle_size, criterion, lambda1, num_epoches):
  batch_size = 64
  auto = autoencoder(hidden_train.shape[1], middle_size).to(device)
  auto_optimizer = optim.Adam(auto.parameters(), lr = 0.001)
  auto_scheduler = optim.lr_scheduler.LambdaLR(auto_optimizer, lr_lambda=lambda1)
  criterion_auto = nn.MSELoss()

  autodataset = MyAutoDataset(hidden_train, label_train)
  trainloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

  autodataset = MyAutoDataset(hidden_eval, label_eval)
  evalloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

  for epoch in range(num_epoches):
    for (data, label) in trainloader:
      # img, _ = data
      # img = img.view(img.size(0), -1)
      # img = Variable(img).cuda()
      #data = torch.tensor(data.clone().detach(), dtype=torch.float).to(device)
      # ===================forward=====================
      data = data.to(device)
      output, _ = auto(data)
      loss = criterion(output, data)
      # ===================backward====================
      auto_optimizer.zero_grad()
      loss.backward()
      auto_optimizer.step()
      auto_scheduler.step()
  # ===================log========================
    for (data, label) in evalloader:
      data = data.to(device)
      # ===================forward=====================
      output, _ = auto(data)
      loss_eval = criterion(output, data)
    # if epoch % 200 == 0:
    #   print('epoch [{}/{}], train loss:{:.4f} eval loass:{:.4f}'
    #         .format(epoch + 1, num_epoches, loss.item(), loss_eval.item()))
      
   ## extract hidden train
  count = 0
  for (data, label) in trainloader:  
    data = data.to(device)
    _, encoder_output = auto(data)

    if count == 0:
      np_out_train = encoder_output.detach().cpu().numpy()
      label_train = label
    else:
      label_train = np.hstack((label_train, label))
      np_out_train = np.vstack((np_out_train, encoder_output.detach().cpu().numpy())) 
    count += 1
  
  ## extract hidden eval
  count = 0
  for (data, label) in evalloader:
    data = data.to(device)
    _, encoder_output = auto(data)

    if count == 0:
      np_out_eval = encoder_output.detach().cpu().numpy()
      label_eval = label

    else:
      label_eval = np.hstack((label_eval, label))
      np_out_eval = np.vstack((np_out_eval, encoder_output.detach().cpu().numpy()))
    count += 1
 
  return np_out_train, np_out_eval, label_train, label_eval

def clustering_knn_acc(model, train_loader, eval_loader, criterion , num_epoches = 400, middle_size = 125):
    hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader)
    #print(hi_train.shape)

    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 50)
    np_out_train, np_out_eval, au_l_train, au_l_eval = train_autoencoder(hi_train, hi_eval, label_train,
                      label_eval, middle_size, criterion, lambda1, num_epoches)


       # print(hi_train.shape)
    knn_acc_1 = knn(hi_train, hi_eval, label_train, label_eval, nn=1)
    knn_acc_au = knn(np_out_train, np_out_eval, au_l_train, au_l_eval, nn=1)
    return knn_acc_1, knn_acc_au


def training(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, file_output,
             root_path, network, en_num_layers, hidden_size, num_class=10,
             few_knn=False):
    auto_criterion = nn.MSELoss()
    start = time.time()
    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 50)
    model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    past_acc = 0

    for ith_epoch in range(1, epoch + 1):
        if ith_epoch % print_every == 0 or ith_epoch == 1:
            ave_loss_train = evaluation(train_loader, model, criterion_seq)
            ave_loss_eval = evaluation(eval_loader,  model, criterion_seq)
            knn_acc_1, knn_acc_au = clustering_knn_acc(model, train_loader, eval_loader, criterion= auto_criterion)

            print('%s (%d %d%%) TrainLoss %.4f EvalLoss %.4f KnnACC W/O-AEC: %.4f W-AEC: %.4f' % (
                timeSince(start, ith_epoch / epoch),
                ith_epoch, ith_epoch / epoch * 100, ave_loss_train, ave_loss_eval, knn_acc_1, knn_acc_au))
            file_output.writelines('%.4f %.4f %.4f %.4f\n' %
                                   (ave_loss_train, ave_loss_eval, knn_acc_1,
                                    knn_acc_au))

            if knn_acc_1 > past_acc:
                past_acc = knn_acc_1
                for item in os.listdir(root_path + 'seq2seq_model/'):
                    if item.startswith('%slayer%d_hid%d' % (network, en_num_layers, hidden_size)):
                        open(root_path + 'seq2seq_model/' + item, 'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                        os.remove(root_path + 'seq2seq_model/' + item)

                path_model = root_path + 'seq2seq_model/%slayer%d_hid%d_epoch%d' % (network, en_num_layers, hidden_size, ith_epoch)
                save_checkpoint(model, epoch, optimizer, ave_loss_train, path_model)
       
        for it, (data, seq_len, label) in enumerate(train_loader):
            input_tensor = data.to(device)
            total_loss, en_hid = train_iter(input_tensor, seq_len, model, optimizer, criterion_seq)

        if ith_epoch % 50 == 0:
          filename = file_output.name 
          file_output.close()
          file_output = open(filename, 'a')

    return total_loss, en_hid