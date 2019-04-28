import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable

REWARD_WEIGHTS = {'dscr': 1, 'mle': 0.2}


class PolicyUpdate(nn.Module):
    """ 
    "Reward-Refined NLLLoss Function for adversarial reinforcement training of generator"
    Really, performing the policy update for the REINFORCE algorithm + mods
    """
    def __init__(self, use_cuda, **kwargs):
        self.use_cuda = use_cuda
        super().__init__(**kwargs)

    def forward(self, probs, targets, dscr_rewards, mle_rewards):
        """
        Args:
            probs: (seq_len, vocab_size) - torch Variable
            targets: (seq_len) - torch Variable
            rewards: (seq_len) - torch Variable
        """
        _, _, vocab_size = probs.size()
        probs = probs.view((-1, vocab_size))
        one_hot = torch.zeros(probs.size())
        indices = targets.data.view((-1, 1))
        # rewards = rewards.data.view((-1, 1))
        if self.use_cuda and torch.cuda.is_available(): 
            one_hot = one_hot.cuda()   
            indices = indices.cuda()
        # write 1 into all positions specified by targets in the 1st dim
        one_hot.scatter_(1, indices, 1) 
        one_hot = Variable(one_hot.type(torch.ByteTensor))  # sets the type, so it can be used in masked_select
        if self.use_cuda and torch.cuda.is_available():
            one_hot = one_hot.cuda()
        policy_probs = torch.masked_select(probs, one_hot)

        dscr_rewards = dscr_rewards.data.view((-1))
        mle_rewards = mle_rewards.data.view((-1))
        rewards = REWARD_WEIGHTS['dscr'] * dscr_rewards + REWARD_WEIGHTS['mle'] * mle_rewards

        return policy_probs, -torch.dot(policy_probs, rewards)
