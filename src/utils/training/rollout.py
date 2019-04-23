"""
Calculate Rewards by 'rolling out' generated sequences to get an idea of state
"""
import copy
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from collections import deque

import pdb

NUM_PROCESSES = 4

class Rollout(object):
    """
    Monte-Carlo Rollout Policy
    """
    def __init__(self, model, update_rate, device):
        self.model = model
        self.rollout_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.device = device # something something data parallel

    def _get_reward_process(self, rollout_num, data_subseqs, discriminator, seq_len, rewards):
        samples = self.rollout_model.sample(batch_size, seq_len, data_subseqs)
        # samples = self.model.sample(batch_size, seq_len, data_subseqs)
        if torch.cuda.is_available():
            samples = samples.cuda()
        samples.to(self.device)

        pred = discriminator(samples)
        pred = pred.cpu().data[:, 1].numpy() # why cpu?
        if rollout_num == 0:
            rewards.append(pred)
        else:
            rewards[l - 1] += pred

    def get_reward(self, data, rollout_num, discriminator):
        """
        args:
            data: input data (batch_size, seq_len)
            rollout_num: roll-out number
            discriminator: discriminator model

            For `rollout_num` iterations, rollout the sequence from each timestep in order to get an idea of the
            generator's state at that step. Rewards for each step equal the average of the discriminator's predicted
            likelihood that the `rollout_num` rollouts from those steps are real.
        """
        rewards = []
        batch_size, seq_len = data.size()
        discriminator.share_memory() # required for multiprocessing
        processes = deque()
        num_procs = 0
        # for data in tqdm(data_iter, desc=' - Create Real Data File', leave=False):
        for i in range(rollout_num):
            print("Rollout #%d: " % i)
            for l in tqdm(range(1, seq_len + 1), desc=' - Rollout for Generator Training'):
                data_subseqs = data[:, :l]
                if num_procs > NUM_PROCESSES:
                    processes[0].join()
                    processes.popleft()
                p = mp.Process(target=self._get_reward_process, args=(i, data_subseqs, discriminator, seq_len, rewards))
                p.start()
                processes.append(p)
                num_procs += 1
        for p in processes:
            p.join()

        rewards = np.transpose(np.array(rewards)) / float(rollout_num)
        return rewards

    def update_params(self):
        """
        With the exception of the embedding layers, which are transferred in directly from the actual generator
        being trained in the adversarial loop, update the weights of the copy model via a weighted average of the
        actual generator's current params and the copy model's current params. This, in essence, moves the copy
        model at a slower rate in the same direction as the generator is moving.
        """
        dic = {}
        for name, param in self.model.named_parameters():
            dic[name] = param.data
        for name, param in self.rollout_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
        # for name, param in self.model.named_parameters():
        #     if name.startswith('embed'):
        #         continue
        #     else:
        #         param.data = self.update_rate * param.data + (1 - self.update_rate) * param.data
