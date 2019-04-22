"""
Calculate Rewards by 'rolling out' generated sequences to get an idea of state
"""
import copy
import numpy as np
import pdb
import torch
import torch.multiprocessing as mp
from collections import deque
from tqdm import tqdm

from .. import constants as const

MAX_NUM_PROCESSES = 8
mp.set_start_method('spawn', force=True)


class Rollout(object):
    """
    Monte-Carlo Rollout Policy
    """
    def __init__(self, generator, update_rate, device):
        self.generator = generator
        self.rollout_generator = copy.deepcopy(generator)
        self.update_rate = update_rate
        self.device = device  # something something data parallel

    def get_rewards(self, data, chord_roots, chord_types, num_rollouts, discriminator):
        """
        args:
            data: input data (batch_size, seq_len)
            chord_roots: input data (batch_size, seq_len)
            chord_types: input data (batch_size, seq_len)
            rollout_num: number of roll-outs
            discriminator: discriminator generator

            For `rollout_num` iterations, rollout the sequence from each timestep in order to get an idea of the
            generator's state at that step. Rewards for each step equal the average of the discriminator's predicted
            likelihood that the `rollout_num` rollouts from those steps are real.
        """
        # required for multiprocessing
        self.rollout_generator.share_memory()
        discriminator.share_memory()  
        pool = mp.Pool(processes=4)

        rewards = []
        batch_size, seq_len = data.size()
        # ---- debugging ----
        # return np.ones(data.shape) * 0.5
        # -------------------
        for rn in range(num_rollouts):
            print("Rollout #%d: " % rn)
            args_list = []
            for ss_idx in range(1, seq_len + 1):
                # Add process arguments
                data_subseqs = data[:, :ss_idx]
                args_list.append((ss_idx, discriminator, batch_size, seq_len, data_subseqs, chord_roots, chord_types))

            results = pool.imap(self._get_reward_process, args_list, chunksize=20)
            for (ss_idx, pred) in tqdm(results, desc=' - Rollout for Generator Training'):
                if rn == 0:
                    rewards.append(pred)
                else:
                    rewards[ss_idx - 1] += pred

        rewards = np.transpose(np.array(rewards)) / float(num_rollouts)
        return rewards

    def _get_reward_process(self, args):
        subseq_idx, discriminator, batch_size, seq_len, data_subseqs, chord_roots, chord_types = args
        samples = self.rollout_generator.sample(batch_size, seq_len, chord_roots, chord_types, seed=data_subseqs)
        if torch.cuda.is_available():
            samples = samples.cuda()
        samples.to(self.device)

        pred = discriminator(samples.cpu(), chord_roots.cpu(), chord_types.cpu())
        pred = pred.cpu().data[:, 1].numpy()  # why cpu?
        return subseq_idx, pred

    def update_params(self):
        """
        With the exception of the embedding layers, which are transferred in directly from the actual generator
        being trained in the adversarial loop, update the weights of the copy generator via a weighted average of the
        actual generator's current params and the copy generator's current params. This, in essence, moves the copy
        generator at a slower rate in the same direction as the generator is moving.
        """
        dic = {}
        for name, param in self.generator.named_parameters():
            dic[name] = param.data
        for name, param in self.rollout_generator.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
        # for name, param in self.generator.named_parameters():
        #     if name.startswith('embed'):
        #         continue
        #     else:
        #         param.data = self.update_rate * param.data + (1 - self.update_rate) * param.data
