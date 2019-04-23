import os.path as op
import pickle as pkl
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from functools import partial
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_iter import DscrDataset
from .subset_dataloader import SubsetDataloaderFactory
from .. import constants as const
from .. import helpers as hlp


class DiscriminatorPretrainer:

    def __init__(self, discriminator, dataset, cache_dir, temp_data_dir, device, args):
        self.discriminator = discriminator
        self.sdlf = SubsetDataloaderFactory(dataset)
        self.cache_dir = cache_dir
        self.device = device
        self.args = args

        if not op.exists(self.cache_dir):
            op.makedirs(self.cache_dir)

        if not op.exists(temp_data_dir):
            op.makedirs(temp_data_dir)
        self.real_data_path = op.join(temp_data_dir, 'real_data.pkl')
        self.gen_data_path = op.join(temp_data_dir, 'generated_data.pkl')

        self._create_real_data_file = partial(hlp.create_real_data_file, outpath=self.real_data_path)
        self._create_generated_data_file = partial(hlp.create_generated_data_file, outpath=self.gen_data_path,
                                                   cuda=self.args.cuda, device=self.device)

        self.criterion = nn.NLLLoss(size_average=False)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=self.args.dscr_learning_rate)
        if self.args.cuda and torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

    def train(self, generator):
        losses = []
        for i in range(const.DSCR_PRETRAIN_DATA_GENS):
            train_data_iter = self._get_dscr_data_iter(generator, num_samples=const.NUM_TRAIN_SAMPLES)

            for j in range(const.DSCR_PRETRAIN_EPOCHS):
                loss = self._train_epoch(train_data_iter)
                losses.append(loss)
                print("::PRETRAIN DSCR:: Data Gen [%d] Epoch [%d] Loss: %f" % (i, j, loss))

            # eval_data_iter = self._get_dscr_data_iter(generator, num_samples=const.NUM_EVAL_SAMPLES)
            # acc = self._eval_epoch(eval_data_iter)

        print('Caching Pretrained Discriminator ...')
        torch.save({'state_dict': self.discriminator.state_dict(),
                    'data_gens': const.DSCR_PRETRAIN_DATA_GENS,
                    'epochs_per_gen': const.DSCR_PRETRAIN_EPOCHS,
                    'loss': loss,
                    'datetime': datetime.now().isoformat()}, op.join(self.cache_dir, 'discriminator.pt'))

        torch.save({'losses': losses}, op.join(self.cache_dir, 'discriminator_losses.pt'))

    def _train_epoch(self, data_iter):
        # trains `model` for one epoch using data from `data_iter`. 
        total_loss = 0.0
        total_batches = 0.0
        for (data, target) in tqdm(data_iter, desc=' - Training Discriminator', leave=False):
            seq, cr, ct = data
            seq_var, cr_var, ct_var, target_var = self.prepare_vars(seq, cr, ct, target)

            target_var = target_var.contiguous().view(-1)
            pred = self.discriminator.forward(seq_var, cr_var, ct_var)
            pred = pred.view(-1, pred.size()[-1])

            loss = self.criterion(pred, target_var)
            total_loss += loss.item()
            total_batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / (total_batches * const.BATCH_SIZE)

    def _eval_epoch(self, data_iter):
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for (data, target) in tqdm(data_iter, desc=" - Evaluation", leave=False):
                seq, cr, ct = data
                data_var, cr_var, ct_var = Variable(seq), Variable(cr), Variable(ct)
                target_var = Variable(target)

                if self.args.cuda and torch.cuda.is_available():
                    data_var, cr_var, ct_var, target_var = data_var.cuda(), cr_var.cuda(), ct_var.cuda(), target_var.cuda()

                data_var = data_var.to(self.device)
                cr_var = cr_var.to(self.device)
                ct_var = ct_var.to(self.device)
                target_var = target_var.to(self.device)

                target_var = target_var.contiguous().view(-1)
                pred = self.discriminator.forward(data_var, cr_var, ct_var)
                pred = pred.view(-1, pred.size()[-1])

                loss = self.criterion(pred, target_var)
                total_loss += loss.item()
                total_batches += 1

                # ---- debugging ----
                # total_batches += 1
                # if total_batches > 100:
                #   break
                # -------------------

        return total_loss / total_batches * const.BATCH_SIZE

    def _get_dscr_data_iter(self, generator, num_samples):
        data_iter = self.sdlf.get_subset_dataloader(num_samples)
        self._create_generated_data_file(generator, data_iter)
        self._create_real_data_file(data_iter)
        dscr_data_iter = DataLoader(DscrDataset(self.real_data_path, self.generated_data_path), batch_size=const.BATCH_SIZE, shuffle=True)
        return dscr_data_iter
