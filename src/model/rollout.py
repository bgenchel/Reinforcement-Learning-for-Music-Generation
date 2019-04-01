"""
Calculate Rewards by 'rolling out' generated sequences to get an idea of state
"""
import copy
import numpy as np

class Rollout(object):
    """
    Monte-Carlo Rollout Policy
    """
    def __init__(self, model, update_rate):
        self.model = model
        self.rollout_model = copy.deepcopy(model)
        self.update_rate = update_rate

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
        for i in range(rollout_num):
            for l in range(1, seq_len + 1):
                data_subseqs = data[:, :l]
                samples = self.rollout_model.sample(batch_size, seq_len, data_subseqs)
                # samples = self.model.sample(batch_size, seq_len, data_subseqs)
                pred = discriminator(samples)
                pred = pred.cpu().data[:, 1].numpy() # why cpu?
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

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
