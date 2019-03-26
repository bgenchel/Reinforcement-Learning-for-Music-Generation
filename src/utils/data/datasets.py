import h5py
import os.path as op
import sys
import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
import constants as const


class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        if not op.exists(h5_path):
            raise Exception("HDF5Dataset:: %s does not exist." % h5_path)
        self.h5 = h5py.File(h5_path, 'r')
        self.h5_keys = list(self.h5.keys())
        self.length = self.h5['sequences'].shape[0]
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seqs = {k: torch.LongTensor(self.h5[k][index]) for k in list(self.h5.keys())}
        return seqs
    
    def terminate(self):
        self.h5.close()
