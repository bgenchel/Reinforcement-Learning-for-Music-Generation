"""
taken from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import constants as const


class Discriminator(nn.Module):
    """
    A CNN Model that gives a softmax output over 2 classes, which is taken to be the likelihood of being real or fake
    for an input sequence.
    """
    def __init__(self, vocab_size, embed_dim, filter_lengths, 
                 num_filters, output_dim, dropout=0.0, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.embedder = nn.Embedding(vocab_size, embed_dim)

        self.chord_root_embedder = nn.Embedding(const.CHORD_ROOT_DIM, const.CHORD_ROOT_EMBED_DIM)
        self.chord_type_embedder = nn.Embedding(const.CHORD_TYPE_DIM, const.CHORD_TYPE_EMBED_DIM)

        chord_dim = const.CHORD_ROOT_EMBED_DIM + const.CHORD_TYPE_EMBED_DIM
        self.chord_encoder = nn.Sequential(
            nn.Linear(chord_dim, (const.CHORD_EMBED_DIM + chord_dim) // 2),
            nn.ReLU(),
            nn.Linear((const.CHORD_EMBED_DIM + chord_dim) // 2, const.CHORD_EMBED_DIM)
        )

        seq_height = embed_dim + const.CHORD_EMBED_DIM
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, n, (length, seq_height)) for (n, length) in zip(num_filters, filter_lengths)
        ])

        self.fc1 = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(sum(num_filters), output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            # note to self, look up why this range is chosen
            param.data.uniform_(-0.05, 0.05)

    def forward(self, x, chord_roots, chord_types):
        """
        Args:
            x: (batch_size * seq_len)
        """
        # dims: batch_size * seq_len * cr_embed_dim
        cr_embeds = self.chord_root_embedder(chord_roots)
        # dims: batch_size * seq_len * ct_embed_dim
        ct_embeds = self.chord_type_embedder(chord_types)
        # dims: batch_size * 1 * seq_len * chord_encode_dim
        chord_encode = self.chord_encoder(torch.cat([cr_embeds, ct_embeds], 2)).unsqueeze(1)
        # dims: batch_size * 1 * seq_len * embed_dim
        embedded = self.embedder(x).unsqueeze(1)
        # dims: batch_size * 1 * seq_len * embed_dim + chord_encode_dim
        full_vec = torch.cat([embedded, chord_encode], 3)
        # dims: [batch_size * num_filter * length]
        convs = [F.relu(conv_layer(full_vec)).squeeze(-1) for conv_layer in self.conv_layers]
        # dims: [batch_size * num_filter]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        pred = self.fc1(pred)
        pred = self.sigmoid(pred) * F.relu(pred) + (1. - self.sigmoid(pred)) * pred
        pred = self.logsoftmax(self.fc2(self.dropout(pred)))
        return pred


# class Discriminator2(nn.Module):
#     """
#     A CNN Model that gives a softmax output over 2 classes, which is taken to be the likelihood of being real or fake
#     for an input sequence.
#     """
#     def __init__(self, vocab_size, embed_dim, filter_lengths, 
#                  num_filters, output_dim, dropout=0.0, **kwargs):
#         super(Discriminator2, self).__init__(**kwargs)
#         self.embedder = nn.Embedding(vocab_size, embed_dim)

#         self.chord_root_embedder = nn.Embedding(const.CHORD_ROOT_DIM, const.CHORD_ROOT_EMBED_DIM)
#         self.chord_type_embedder = nn.Embedding(const.CHORD_TYPE_DIM, const.CHORD_TYPE_EMBED_DIM)

#         chord_dim = const.CHORD_ROOT_EMBED_DIM + const.CHORD_TYPE_EMBED_DIM
#         self.chord_encoder = nn.Sequential(
#             nn.Linear(chord_dim, (const.CHORD_EMBED_DIM + chord_dim) // 2),
#             nn.ReLU(),
#             nn.Linear((const.CHORD_EMBED_DIM + chord_dim) // 2, const.CHORD_EMBED_DIM)
#         )

#         seq_height = embed_dim + const.CHORD_EMBED_DIM
#         self.conv_short = nn.Sequential(
#             nn.Conv2d(1, 8, (const.DUR_TICKS_MAP['32nd'], seq_height)),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, (const.DUR_TICKS_MAP['16th'], 1)),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, (const.DUR_TICKS_MAP['8th'], 1)),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, (const.DUR_TICKS_MAP['quarter'], 1)),
#             nn.ReLU()
#         )

#         self.conv_med = nn.Sequential(
#             nn.Conv2d(1, 8, (const.DUR_TICKS_MAP['8th'], seq_height)),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, (const.DUR_TICKS_MAP['quarter'], 1)),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, (const.DUR_TICKS_MAP['half'], 1)),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, (const.DUR_TICKS_MAP['whole'], 1)),
#             nn.ReLU()
#         )

#         self.conv_long = nn.Sequential(
#             nn.Conv2d(1, 16, (const.DUR_TICKS_MAP['half'], seq_height)),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, (const.DUR_TICKS_MAP['whole'], 1)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (const.DUR_TICKS_MAP['double'], 1)),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, (const.DUR_TICKS_MAP['4-bar'], 1)),
#             nn.ReLU()
#         )

#         self.conv_layer_paths = nn.ModuleList([self.conv_short, self.conv_med, self.conv_long])

#         total_filters = sum([32, 32, 64])
#         self.fc1 = nn.Linear(total_filters, (total_filters + output_dim) // 2)
#         self.dropout = nn.Dropout(p=dropout)
#         self.fc2 = nn.Linear((total_filters + output_dim) // 2, output_dim)
#         self.softmax = nn.LogSoftmax(dim=-1)
#         self.sigmoid = nn.Sigmoid()
#         self.init_parameters()

#     def init_parameters(self):
#         for param in self.parameters():
#             # note to self, look up why this range is chosen
#             param.data.uniform_(-0.05, 0.05)

#     def forward(self, x, chord_roots, chord_types):
#         """
#         Args:
#             x: (batch_size * seq_len)
#         """
#         # dims: batch_size * seq_len * cr_embed_dim
#         cr_embeds = self.chord_root_embedder(chord_roots)
#         # dims: batch_size * seq_len * ct_embed_dim
#         ct_embeds = self.chord_type_embedder(chord_types)
#         # dims: batch_size * 1 * seq_len * chord_encode_dim
#         chord_encode = self.chord_encoder(torch.cat([cr_embeds, ct_embeds], 2)).unsqueeze(1)
#         # dims: batch_size * 1 * seq_len * embed_dim
#         embedded = self.embedder(x).unsqueeze(1)
#         # dims: batch_size * 1 * seq_len * embed_dim + chord_encode_dim
#         full_vec = torch.cat([embedded, chord_encode], 3)
#         # dims: [batch_size * num_filter * length]
#         convs = [conv_layer_path(full_vec).squeeze(-1) for conv_layer_path in self.conv_layer_paths]
#         # dims: [batch_size * num_filter]
#         pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
#         pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
#         pred = self.fc1(pred)
#         pred = self.sigmoid(pred) * F.relu(pred) + (1. - self.sigmoid(pred)) * pred
#         pred = self.softmax(self.fc2(self.dropout(pred)))
#         return pred
