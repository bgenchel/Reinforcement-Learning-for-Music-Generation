import os
import os.path as op
import pdb
import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from functools import partial
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from .gan_loss import GANLoss
from .rollout import Rollout
from .data_iter import DscrDataset
from .subset_dataloader import SubsetDataloaderFactory
from .. import constants as const
from .. import helpers as hlp


class AdversarialRLTrainer:

    def __init__(self, generator, discriminator, dataset, temp_data_dir, valid_iter, device, args):
        self.generator = generator
        self.discriminator = discriminator
        self.valid_iter = valid_iter
        self.device = device
        self.args = args

        self.prepare_vars = partial(hlp.prepare_vars, self.args.cuda, self.device)

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
            os.makedirs(temp_data_dir)
        
        self.real_data_path = op.join(temp_data_dir, 'real.data')
        self.gen_data_path = op.join(temp_data_dir, 'generated.data')

        self._create_real_data_file = partial(hlp.create_real_data_file, outpath=self.real_data_path)
        self._create_generated_data_file = partial(hlp.create_generated_data_file, outpath=self.gen_data_path,
                                                   cuda=self.args.cuda, device=self.device)

        self.run_dir = op.join("runs", self.args.dataset, datetime.now().strftime('%b%d-%y_%H:%M:%S'))
        if not op.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.gen_losses = None
        self.dscr_losses = None

    def train(self):
        self.gen_losses = []
        self.dscr_losses = []
        min_gen_loss = float('inf') 
        for epoch in range(const.ADV_TRAIN_EPOCHS): 
            print("#" * 20 + "\n Adv Epoch[%d]\n" % epoch + "#" * 20) 
            print("-" * 5 + ' Generator Training ' + '-' * 5)
            gen_loss = self._train_generator(epoch)
            self.gen_losses.append(gen_loss)

            if (gen_loss) < min_gen_loss:
                self._save_models()
                self._save_losses()

            # Train the Discriminator for `D_STEPS` each on `D_DATA_GENS` sets of generated and sampled real data
            print("-" * 5 + ' Discriminator Training ' + '-' * 5)
            dscr_loss = self._train_discriminator(epoch)
            self.dscr_losses.append(dscr_loss)

        self._save_models()
        self._save_losses()

    def _train_generator(self, epoch):
        sdl = self.sdlf.get_subset_dataloader(const.G_STEPS * const.BATCH_SIZE)
        total_loss = 0.0
        # Train the generator for G_STEPS
        # for gstep in range(const.G_STEPS):
        for gstep, data in enumerate(sdl):
            # Generate a batch of sequences
            # samples = self.generator.module.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN)
            chord_roots, chord_types = data[const.CR_SEQS_KEY], data[const.CT_SEQS_KEY]
            # TODO: switch out the constants for params from cr and ct
            samples = self.generator.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN, chord_roots, chord_types)

            # Calculate rewards for generator state using monte carlo rollout
            # rewards are equal to the average (over all rollouts) of the discriminator's likelihood of realism score
            rewards = self.rollout.get_rewards(samples, chord_roots, chord_types, const.NUM_ROLLOUTS, self.discriminator.cpu())
            if self.args.cuda and torch.cuda.is_available():
                self.discriminator = self.discriminator.cuda()

            # Because the discriminator gives results in negative log likelihood, exponent it
            rewards = torch.exp(torch.Tensor(rewards)).contiguous().view((-1,))
            rewards = self.prepare_vars(rewards)

            # Construct a corresponding input step for the sequences: 
            # add a timestep of zeros before samples and then delete its last column.
            zeros = torch.zeros((const.BATCH_SIZE, 1)).type(torch.LongTensor)
            cr_zeros = torch.zeros((const.BATCH_SIZE, 1)).type(torch.LongTensor)
            ct_zeros = torch.zeros((const.BATCH_SIZE, 1)).type(torch.LongTensor)
            zeros, cr_zeros, ct_zeros = self.prepare_vars(zeros, cr_zeros, ct_zeros)

            # Make them cuda
            chord_roots, chord_types = self.prepare_vars(chord_roots, chord_types)

            inputs = torch.cat([zeros, samples], dim=1)[:, :-1].contiguous()
            cr_inputs = torch.cat([cr_zeros, chord_roots], dim=1)[:, :-1].contiguous()
            ct_inputs = torch.cat([ct_zeros, chord_types], dim=1)[:, :-1].contiguous()
            inputs, cr_inputs, ct_inputs, targets = self.prepare_vars(inputs, cr_inputs, ct_inputs, samples)

            # Make a forward prediction of the next step given the inputs using the generator
            prob = self.generator.forward(inputs, cr_inputs, ct_inputs)
            adv_loss = self.gan_loss(prob, targets, rewards) / const.BATCH_SIZE  # from suragnair/seqGAN
            total_loss += adv_loss
            print('Adv Epoch [%d], Gen Step [%d] - Train Loss: %f' % (epoch, gstep, adv_loss))
            self.gan_loss.zero_grad()
            adv_loss.backward()
            self.gan_optimizer.step()

            # Update the parameters of the generator copy inside the rollout object.
            self.rollout.update_params()

            # Check how our MLE validation changes with GAN loss. We've noticed it going up, but are unsure
            # if this is a good metric by which to validate for this type of training.
            # valid_loss = self._eval_epoch(self.generator, self.valid_iter, self.eval_loss)
            # print('Adv Epoch [%d], Gen Step [%d] - Valid Loss: %f' % (epoch, gstep, valid_loss))

        return total_loss / const.G_STEPS

    def _train_discriminator(self, epoch):
        total_dscr_loss = 0.0
        for data_gen in range(const.D_DATA_GENS):
            data_iter = self._get_dscr_data_iter(const.NUM_TRAIN_SAMPLES)
            for dstep in range(const.D_STEPS):
                loss = self._train_dscr_epoch(data_iter) 
                total_dscr_loss += loss
                print('Adv Epoch [%d], Dscr Gen [%d], Dscr Step [%d] - Loss: %f' % (epoch, data_gen, dstep, loss))
        return total_dscr_loss / (const.D_DATA_GENS * const.D_STEPS)

    def _train_dscr_epoch(self, data_iter):
        total_loss = 0.0
        total_batches = 0.0
        for (data, target) in tqdm(data_iter, desc=' - AdversarialRL Train Discriminator', leave=False):
            seq, cr, ct = data
            seq_var, cr_var, ct_var, target_var = self.prepare_vars(seq, cr, ct, target)

            target_var = target_var.contiguous().view(-1)
            pred = self.discriminator.forward(seq_var, cr_var, ct_var)
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
        # total_batches = 0  # temp for debugging
        with torch.no_grad():
            for data in tqdm(data_iter, desc=" - Generator Evaluation", leave=False):
                data_var = data[const.SEQS_KEY]
                cr_var = data[const.CR_SEQS_KEY]  # chord roots
                ct_var = data[const.CT_SEQS_KEY]  # chord types
                target_var = data[const.TARGETS_KEY]

                if self.args.cuda and torch.cuda.is_available():
                    data_var, cr_var, ct_var, target_var = data_var.cuda(), cr_var.cuda(), ct_var.cuda(), target_var.cuda()

                data_var = data_var.to(self.device)
                cr_var = cr_var.to(self.device)
                ct_var = ct_var.to(self.device)
                target_var = target_var.to(self.device)

                target_var = target_var.contiguous().view(-1)
                pred = self.generator.forward(data_var, cr_var, ct_var)
                pred = pred.view(-1, pred.size()[-1])

                loss = self.eval_loss(pred, target_var)
                total_loss += loss.item()
                total_words += data_var.size(0) * data_var.size(1)

                ############################
                # just temporary, for debugging
                # total_batches += 1
                # if total_batches > 10:
                #   break
                ###########################

        return total_loss / total_words

    def _get_dscr_data_iter(self, num_samples):
        data_iter = self.sdlf.get_subset_dataloader(num_samples)
        self._create_generated_data_file(self.generator, data_iter)
        self._create_real_data_file(data_iter)
        dscr_data_iter = DataLoader(DscrDataset(self.real_data_path, self.gen_data_path), batch_size=const.BATCH_SIZE, shuffle=True)
        return dscr_data_iter

    def _save_models(self):
        """
        Save the model and parameters needed to reinstantiate it at the end of training 
        """
        print("Saving models & losses ... ")
        gen_model_inputs = {'vocab_size': const.VOCAB_SIZE,
                            'embed_dim': const.GEN_EMBED_DIM,
                            'hidden_dim': const.GEN_HIDDEN_DIM,
                            'use_cuda': self.args.cuda}

        torch.save({'state_dict': self.generator.state_dict(),
                    'model_inputs': gen_model_inputs}, op.join(self.run_dir, 'generator.pt'))

        dscr_model_inputs = {'vocab_size': const.VOCAB_SIZE, 
                             'embed_dim': const.DSCR_EMBED_DIM,
                             'filter_sizes': const.DSCR_FILTER_SIZES,
                             'num_filters': const.DSCR_NUM_FILTERS,
                             'output_dim': const.DSCR_NUM_CLASSES,
                             'dropout': const.DSCR_DROPOUT}

        torch.save({'state_dict': self.discriminator.state_dict(),
                    'model_inputs': dscr_model_inputs}, op.join(self.run_dir, 'discriminator.pt'))

    def _save_losses(self):
        torch.save({'adv_gen_losses': self.gen_losses, 'adv_dscr_losses': self.dscr_losses}, op.join(self.run_dir, 'losses.pt'))
