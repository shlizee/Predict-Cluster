from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler

from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

import random
import torch

import h5py
import numpy as np

def get_data_list(data_path):
    f = h5py.File(data_path, 'r')
    data_list = []
    label_list = []
    for i in range(len(f['label'])):

        if np.shape(f[str(i)][:])[0] > 10:
            x = f[str(i)][:]
            # original matrix with probability
            y = f['label'][i]

            x = torch.tensor(x, dtype=torch.float)

            data_list.append(x)
            label_list.append(y)

    return data_list, label_list

def concate_data(data_path, seq_len = 10):
    data_list, label_list = get_data_list(data_path)

    feature_len = data_list[0].size()[-1]
    data = torch.tensor(())
    for i in range(len(label_list)):
        if data_list[i].size()[0] == seq_len:
            tmp = troch.flatten(data_list[i])
            data = torch.cat((data, tmp)).unsqueeze(0) 

        if data_list[i].size()[0] < seq_len:
          dif = seq_len - data_list.size()[0]
          tmp = torch.cat((data_list[i], torch.zeros((dif, feature_len))))
          tmp = torch.flatten(tmp)
          data = torch.cat((data, tmp)).unsqueeze(0) 
        
        if data_list[i].size()[0] > seq_len:
          tmp = data_list[i][:seq_len,:]
          tmp = torch.flatten(tmp).unsqueeze(0) 
          data = torch.cat((data, tmp))
    label_list = np.asarray(label_list)
    return data.numpy(), label_lists


def pad_collate(batch):
    lens = [len(x[0]) for x in batch]

    data = [x[0] for x in batch]
    label = [x[1] for x in batch]
    label = np.asarray(label)
    
    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
    return xx_pad, lens, label



class MyAutoDataset(Dataset):
    def __init__(self, data, label):
      
        self.data = data
        self.label = label
        #self.xy = zip(self.data, self.label)


    def __getitem__(self, index):
        sequence = self.data[index, :]
        label = self.label[index]
        # Transform it to Tensor
        #x = torchvision.transforms.functional.to_tensor(sequence)
        #x = torch.tensor(sequence, dtype=torch.float)
        #y = torch.tensor([self.label[index]], dtype=torch.int)
        
        return sequence, label

    def __len__(self):
        return len(self.label)

class MyDataset(Dataset):
    def __init__(self, data_path):

        self.data, self.label = get_data_list(data_path)

        label = np.asarray(self.label)
        train_index = np.zeros(len(self.label))

    def __getitem__(self, index):
        sequence = self.data[index]
        label = self.label[index]

        return sequence, label

    def __len__(self):
        return len(self.label)