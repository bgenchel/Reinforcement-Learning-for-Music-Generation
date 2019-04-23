import os
import os.path as op
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader

from .gan_loss import GANLoss
from .rollout import Rollout
from .data_iter import DscrDataset
from .subset_dataloader import SubsetDataloaderFactory

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const


class AdversarialRLTrainer:

    def __init__(self, generator, discriminator, dataset, temp_data_dir, valid_iter, device, args):
        self.generator = generator
        self.discriminator = discriminator
        self.valid_iter = valid_iter
        self.device = device
        self.args = args

        self.sdlf = SubsetDataloaderFactory(dataset)
        self.rollout = Rollout(self.generator, update_rate=0.8, device=self.device)

        self.gan_loss = GANLoss(use_cuda=self.args.cuda)
        self.gan_optimizer = optim.Adam(self.generator.parameters(), lr=args.adv_gen_learning_rate)

        self.dscr_loss = nn.NLLLoss(size_average=False)
        self.dscr_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.adv_dscr_learning_rate)

        self.eval_loss = nn.NLLLoss(size_average=False)

        if self.args.cuda and torch.cuda.is_available():
            self.gan_loss = self.gan_loss.cuda()
            self.dscr_loss = self.dscr_loss.cuda()
            self.eval_loss = self.eval_loss.cuda()

        if not op.exists(temp_data_dir):
            op.makedirs(temp_data_dir)
        self.real_data_path = op.join(temp_data_dir, 'real.data')
        self.generated_data_path = op.join(temp_data_dir, 'generated.data')

        self.run_dir = op.join("runs", self.args.dataset, datetime.now().strftime('%b%d-%y_%H:%M:%S'))
        if not op.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.gen_losses = None
        self.dscr_losses = None

    def train(self):
        self.gen_losses = []
        self.dscr_losses = []
        min_gen_loss = float('-inf') 
        for epoch in range(const.GAN_TRAIN_EPOCHS): 
            print("#" * 30 + "\nAdversarial Epoch [%d]\n" % epoch + "#"*30) 
            gen_loss = self._train_generator(epoch)
            self.gen_losses.append(gen_loss)
            # Save the model and parameters needed to reinstantiate it at the end of training 
            if (gen_loss) < min_gen_loss:
                model_inputs = {'vocab_size': const.VOCAB_SIZE,
                                'embed_dim': const.GEN_EMBED_DIM,
                                'hidden_dim': const.GEN_HIDDEN_DIM,
                                'use_cuda': self.args.cuda}

                torch.save({'state_dict': self.generator.state_dict(),
                            'model_inputs': model_inputs}, op.join(self.run_dir, 'generator.pt'))

                torch.save({'adv_gen_losses': self.gen_losses, 'adv_dscr_losses': self.dscr_losses}, op.join(self.run_dir, 'losses.pt'))

            # Train the Discriminator for `D_STEPS` each on `D_DATA_GENS` sets of generated and sampled real data
            dscr_loss = self._train_discriminator(epoch)
            self.dscr_losses.append(dscr_loss)

    def _train_generator(self, epoch):
        total_loss = 0.0
        # Train the generator for G_STEPS
        for gstep in range(const.G_STEPS):
            # Generate a batch of sequences
            samples = self.generator.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN)

            # Construct a corresponding input step for the sequences: 
            # add a timestep of zeros before samples and then delete its last column.
            zeros = torch.zeros((const.BATCH_SIZE, 1)).type(torch.LongTensor)
            if self.args.cuda and torch.cuda.is_available():
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data)

            # Calculate rewards for generator state using monte carlo rollout
            # rewards are equal to the average (over all rollouts) of the discriminator's likelihood of realism score
            rewards = self.rollout.get_reward(samples, const.NUM_ROLLOUTS, self.discriminator)
            rewards = Variable(torch.Tensor(rewards))
            # Because the discriminator gives results in negative log likelihood, exponent it
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if self.args.cuda and torch.cuda.is_available():
                rewards = rewards.cuda()

            # Make a forward prediction of the next step given the inputs using the generator
            prob = self.generator.forward(inputs)
            adv_loss = self.gan_loss(prob, targets, rewards) / const.BATCH_SIZE  # from suragnair/seqGAN
            total_loss += adv_loss
            print('Adv Epoch [%d], Gen Step [%d] - Train Loss: %f' % (epoch, gstep, adv_loss))
            self.optimizer.zero_grad()
            adv_loss.backward()
            self.optimizer.step()

            # Update the parameters of the generator copy inside the rollout object.
            self.rollout.update_params()

            # Check how our MLE validation changes with GAN loss. We've noticed it going up, but are unsure
            # if this is a good metric by which to validate for this type of training.
            valid_loss = self._eval_epoch(self.generator, self.valid_loader, self.eval_loss)
            print('Adv Epoch [%d], Gen Step [%d] - Valid Loss: %f' % (epoch, gstep, valid_loss))

        return total_loss / const.G_STEPS

    def _train_discriminator(self, epoch):
        total_dscr_loss = 0.0
        for data_gen in range(const.D_DATA_GENS):
            data_loader = self._get_dscr_data_iter(self.NUM_TRAIN_SAMPLES)
            self._create_generated_data_file(len(data_loader))
            self._create_real_data_file(data_loader)
            dscr_data_iter = DataLoader(DscrDataset(self.real_data_path, self.generated_data_path), batch_size=const.BATCH_SIZE, shuffle=True)
            for dstep in range(const.D_STEPS):
                loss = self._train_dscr_epoch(self.discriminator, dscr_data_iter, self.dscr_loss, self.dscr_optimizer)
                total_dscr_loss += loss
                print('Adv Epoch [%d], Dscr Gen [%d], Dscr Step [%d] - Loss: %f' % (epoch, data_gen, dstep, loss))
        return total_dscr_loss / (const.D_DATA_GENS * const.D_STEPS)

    def _train_dscr_epoch(self, data_iter):
        total_loss = 0.0
        total_batches = 0.0
        for (data, target) in tqdm(data_iter, desc=' - AdversarialRL Train Discriminator', leave=False):
            data_var = Variable(data)
            target_var = Variable(target)

            if self.args.cuda and torch.cuda.is_available():
                data_var, target_var = data_var.cuda(), target_var.cuda()

            data_var = data_var.to(self.device)
            target_var = target_var.to(self.device)

            target_var = target_var.contiguous().view(-1)
            pred = self.discriminator.forward(data_var)
            pred = pred.view(-1, pred.size()[-1])

            loss = self.dscr_loss(pred, target_var)
            total_loss += loss.item()
            total_batches += 1

            self.dscr_optimizer.zero_grad()
            loss.backward()
            self.dscr_optimizer.step()

        return total_loss / (total_batches * const.BATCH_SIZE)

    def _eval_epoch(self, model, data_iter, loss_fn):
        total_loss = 0.0
        total_words = 0.0
        total_batches = 0
        with torch.no_grad():
            for data in tqdm(data_iter, desc=" - Evaluation", leave=False):
                data_var = Variable(data["sequences"])
                target_var = Variable(data["targets"])

                if self.args.cuda and torch.cuda.is_available():
                    data_var, target_var = data_var.cuda(), target_var.cuda()

                data_var = data_var.to(self.device)
                target_var = target_var.to(self.device)

                target_var = target_var.contiguous().view(-1)
                pred = model.forward(data_var)
                pred = pred.view(-1, pred.size()[-1])

                loss = loss_fn(pred, target_var)
                total_loss += loss.item()
                total_words += data_var.size(0) * data_var.size(1)
                total_batches += 1

        return total_loss / total_words

    def _get_dscr_data_iter(self, num_samples):
        data_iter = self.sdlf.get_subset_dataloader(num_samples)
        self._create_generated_data_file(len(data_iter))
        self._create_real_data_file(data_iter)
        dscr_data_iter = DataLoader(DscrDataset(self.real_data_path, self.generated_data_path), batch_size=const.BATCH_SIZE, shuffle=True)
        return dscr_data_iter

    def _create_generated_data_file(self, num_batches):
        """
        Generates `num_batches` batches of size BATCH_SIZE from the generator. Stores the data in `output_file`
        """
        samples = []
        for _ in tqdm(range(num_batches), desc=" = Create Generated Data File"):
            sample_batch = self.generator.module.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN).cpu().data.numpy().tolist()
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
