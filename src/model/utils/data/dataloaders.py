import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class SplitDataLoader(DataLoader):
    """
    DataLoader that can optionally split the internal dataset and return multiple dataloaders
    """
    def __init__(self, dataset, batch_size=32, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.kwargs = kwargs
        super().__init__(dataset, batch_size=batch_size, **kwargs)

    def split(self, split=0.15, shuffle=True, random_seed=42):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split_size = int(np.floor(split * dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        split1_indices, split2_indices = indices[:-split_size], indices[-split_size:]
        # Creating data samplers and loaders:
        split1_sampler = SubsetRandomSampler(split1_indices)
        split2_sampler = SubsetRandomSampler(split2_indices)

        split1_loader = SplitDataLoader(self.dataset, batch_size=self.batch_size, sampler=split1_sampler, **self.kwargs)
        split2_loader = SplitDataLoader(self.dataset, batch_size=self.batch_size, sampler=split2_sampler, **self.kwargs)        

        return split1_loader, split2_loader
