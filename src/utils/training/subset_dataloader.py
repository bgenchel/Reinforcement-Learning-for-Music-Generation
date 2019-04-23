import os.path as op
import random
import sys
from pathlib import Path
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const

class SubsetDataloaderFactory:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_subset_dataloader(self, num_samples):
        """
        This method allows us to train the model stochastically, as opposed to training over the full dataset, which
        for nottingham is over 170K samples. Each time we need a new real.data file, we first get a new dataloader 
        using this method, that has a new set of NUM_SAMPLES random samples from the original dataset.
        """
        indices = random.sample(range(len(self.dataset)), min(len(self.dataset), num_samples))
        return DataLoader(self.dataset, batch_size=const.BATCH_SIZE, sampler=SubsetRandomSampler(indices), drop_last=True)
