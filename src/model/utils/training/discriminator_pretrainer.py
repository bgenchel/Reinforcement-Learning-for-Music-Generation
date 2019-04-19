import os.path as op
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_iter import DscrDataset
from .subset_dataloader import SubsetDataloaderFactory
from .. import constants as const


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
        self.real_data_path = op.join(temp_data_dir, 'real.data')
        self.generated_data_path = op.join(temp_data_dir, 'generated.data')

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
        # def pretrain_epoch(model, data_iter, loss_fn, optimizer, pretrain_gen=False):
        total_loss = 0.0
        total_batches = 0.0
        for (data, target) in tqdm(data_iter, desc=' - Training Discriminator', leave=False):
            data_var, target_var = Variable(data), Variable(target)

        if self.args.cuda and torch.cuda.is_available():
            data_var, target_var = data_var.cuda(), target_var.cuda()

        data_var = data_var.to(self.device)
        target_var = target_var.to(self.device)

        target_var = target_var.contiguous().view(-1)
        pred = self.discriminator.forward(data_var)
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
        # count = 0  # temp for debugging
        with torch.no_grad():
            for (data, target) in tqdm(data_iter, desc=" - Evaluation", leave=False):
                data_var, target_var = Variable(data), Variable(target)

                if self.args.cuda and torch.cuda.is_available():
                    data_var, target_var = data_var.cuda(), target_var.cuda()

                data_var, target_var = data_var.to(self.device), target_var.to(self.device)

                target_var = target_var.contiguous().view(-1)
                pred = self.discriminator.forward(data_var)
                pred = pred.view(-1, pred.size()[-1])

                loss = self.criterion(pred, target_var)
                total_loss += loss.item()
                total_batches += 1

                ###########################
                # just temporary, for debugging
                # count += 1
                # if count > 100:
                    # break
                ###########################

        return total_loss / total_batches * const.BATCH_SIZE

    def _get_dscr_data_iter(self, generator, num_samples):
        data_iter = self.sdlf.get_subset_dataloader(num_samples)
        self._create_generated_data_file(generator, len(data_iter))
        self._create_real_data_file(data_iter)
        dscr_data_iter = DataLoader(DscrDataset(self.real_data_path, self.generated_data_path), batch_size=const.BATCH_SIZE, shuffle=True)
        return dscr_data_iter

    def _create_generated_data_file(self, generator, num_batches):
        """
        Generates `num_batches` batches of size BATCH_SIZE from the generator. Stores the data in `output_file`
        """
        samples = []
        for _ in tqdm(range(num_batches), desc=" = Create Generated Data File"):
            # sample_batch = generator.module.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN).cpu().data.numpy().tolist()
            sample_batch = generator.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN).cpu().data.numpy().tolist()
            samples.extend(sample_batch)

        with open(self.generated_data_path, 'w') as fout:
            for sample in samples:
                str_sample = ' '.join([str(s) for s in sample])
                fout.write('%s\n' % str_sample)
        return

    def _create_real_data_file(self, data_iter):
        """
        Iterates through `data_iter` and stores all its targets in `output_file`.
        """
        print('creating real data file ...')
        samples = []
        for data in tqdm(data_iter, desc=' - Create Real Data File', leave=False):
            sample_batch = list(data['sequences'].numpy())
            samples.extend(sample_batch)

        with open(self.real_data_path, 'w') as fout:
            for sample in samples:
                str_sample = ' '.join([str(s) for s in sample])
                fout.write('%s\n' % str_sample)
        return
