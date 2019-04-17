import os.path as op
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch.autograd import Variable
from tqdm import tqdm

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const


class GeneratorPretrainer:

    def __init__(self, generator, train_iter, valid_iter, save_dir, device, args):
        self.generator = generator
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.save_dir = save_dir
        self.device = device
        self.args = args

        self.criterion = nn.NLLLoss(size_average=False)
        self.optimizer = optim.Adam(self.generator.parameters(), lr=self.args.gen_learning_rate)
        if self.args.cuda and torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

    def train(self):
        train_losses = []
        valid_losses = []
        min_valid_loss = float('inf')
        for epoch in range(const.GEN_PRETRAIN_EPOCHS):
            train_loss = self._train_epoch()
            train_losses.append(train_loss)
            print('::PRETRAIN GEN:: Epoch [%d] Training Loss: %f' % (epoch, train_loss))

            valid_loss = self._eval_epoch()
            valid_losses.append(valid_loss)
            print('::PRETRAIN GEN:: Epoch [%d] Validation Loss: %f' % (epoch, valid_loss))

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                print('Caching Pretrained Generator ...')
                torch.save({'state_dict': self.generator.state_dict(),
                            'epochs': epoch + 1,
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                            'datetime': datetime.now().isoformat()}, op.join(self.save_dir, 'generator.pt'))

            if epoch > 1:
                return

        torch.save({'train_losses': train_losses,
                    'valid_losses': valid_losses}, op.join(self.save_dir, 'generator_losses.pt'))

    def _train_epoch(self):
        # trains `model` for one epoch using data from `data_iter`. 
        total_loss = 0.0
        total_words = 0.0
        total_batches = 0
        for data in tqdm(self.train_iter, desc=' - Pretraining Generator', leave=False):
            data_var = data["sequences"]
            target_var = data['targets']

            if self.args.cuda and torch.cuda.is_available():
                data_var, target_var = data_var.cuda(), target_var.cuda()

            data_var = data_var.to(self.device)
            target_var = target_var.to(self.device)

            target_var = target_var.contiguous().view(-1)
            pred = self.generator.forward(data_var)
            pred = pred.view(-1, pred.size()[-1])

            loss = self.criterion(pred, target_var)
            total_loss += loss.item()
            total_words += data_var.size(0) * data_var.size(1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_batches += 1
            if total_batches > 10:
                break

        return total_loss / total_words

    def _eval_epoch(self):
        total_loss = 0.0
        total_words = 0.0
        count = 0  # temp for debugging
        with torch.no_grad():
            for data in tqdm(self.valid_iter, desc=" - Generator Evaluation", leave=False):
                data_var = Variable(data["sequences"])
                target_var = Variable(data["targets"])

                if self.args.cuda and torch.cuda.is_available():
                    data_var, target_var = data_var.cuda(), target_var.cuda()

                data_var = data_var.to(self.device)
                target_var = target_var.to(self.device)

                target_var = target_var.contiguous().view(-1)
                pred = self.generator.forward(data_var)
                pred = pred.view(-1, pred.size()[-1])

                loss = self.criterion(pred, target_var)
                total_loss += loss.item()
                total_words += data_var.size(0) * data_var.size(1)

                ############################
                # just temporary, for debugging
                count += 1
                if count > 10:
                    break
                ###########################

        return total_loss / total_words
