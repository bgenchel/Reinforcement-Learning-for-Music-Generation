# -*- coding:utf-8 -*-
import os
import random
import math

import tqdm

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        lis.append(l)
    return lis


class GenDataset(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_file, **kwargs):
        super().__init__(**kwargs)
        self.data_lis = read_file(data_file)

    def __len__(self):
        return len(self.data_lis)

    def __getitem__(self, index):
        return torch.Tensor(np.array(self.data_lis[index]))


class DscrDataset(Dataset):
    """ Toy data iter to load digits"""

    def __init__(self, real_data_file, gen_data_file, **kwargs):
        super().__init__(**kwargs)
        real_data = read_file(real_data_file)
        gen_data = read_file(gen_data_file)
        data = real_data + gen_data
        labels = [1]*len(real_data) + [0]*len(gen_data)
        self.pairs = list(zip(data, labels))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        data, label = self.pairs[index]
        return torch.LongTensor(np.array(data)), torch.LongTensor(np.array([label]))
