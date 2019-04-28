import copy
import os
import os.path as op
import pdb
import torch
import torch.optim as optim
import torch.nn as nn
from colorama import init as colorinit
from colorama import Fore as color
from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from .policy_update import PolicyUpdate
from .rollout import Rollout
from .data_iter import DscrDataset
from .subset_dataloader import SubsetDataloaderFactory
from .. import constants as const
from .. import helpers as hlp

colorinit(autoreset=True)


class AdversarialRLTrainer:

    def __init__(self, generator, discriminator, dataset, temp_data_dir, valid_iter, device, args):
        self.generator = generator
        self.discriminator = discriminator

        self.reward_rnn = copy.deepcopy(self.generator).eval()
        self.reward_rnn.flatten_parameters()

        self.valid_iter = valid_iter
        self.device = device
        self.args = args

        self.prepare_vars = partial(hlp.prepare_vars, self.args.cuda, self.device)

        self.sdlf = SubsetDataloaderFactory(dataset)
        self.rollout = Rollout(self.generator, update_rate=0.8, device=self.device)

        self.policy_update = PolicyUpdate(use_cuda=self.args.cuda)
        self.policy_optimizer = optim.Adam(self.generator.parameters(), lr=args.adv_gen_learning_rate)

        self.dscr_loss = nn.NLLLoss(size_average=False)
        self.dscr_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.adv_dscr_learning_rate)

        self.eval_loss = nn.NLLLoss(size_average=False)

        if self.args.cuda and torch.cuda.is_available():
            self.policy_update = self.policy_update.cuda()
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
        print(color.CYAN + "::INFO:: Saving to %s" % self.run_dir)

        self.gen_losses = None
        self.dscr_losses = None

    def train(self):
        self.gen_updates = []
        self.avg_rewards = []
        self.dscr_losses = []
        self.losses = []
        min_loss = float('inf') 
        try: 
            for epoch in range(const.ADV_TRAIN_EPOCHS): 
                print("#" * 20 + "\n Adv Epoch[%d]\n" % epoch + "#" * 20) 
                print("-" * 10 + ' Generator Training ' + '-' * 10)
                avg_update, avg_reward = self._train_generator(epoch)
                self.gen_updates.append(avg_update)
                self.avg_rewards.append(avg_reward)

                # Train the Discriminator for `D_STEPS` each on `D_DATA_GENS` sets of generated and sampled real data
                print("-" * 10 + ' Discriminator Training ' + '-' * 10)
                dscr_loss = self._train_discriminator(epoch)
                self.dscr_losses.append(dscr_loss)

                # custom loss for tracking quality for saving and intuitive visualization
                # when dscriminator loss is low and rewards are high, loss will be close to 0
                # when discriminator loss is low, loss becomes pretty close to 1 - avg_reward
                # when rewards are low, loss becomes close to dscr_loss
                loss = (1 + dscr_loss) * (1 + (1 - avg_reward)) - 1
                self.losses.append(loss)
                print(color.CYAN + "::INFO:: Adv Loss: {}".format(loss))
                if loss < min_loss:
                    self._save_models()
                    self._save_learning_data()
                    min_loss = loss
        except KeyboardInterrupt:
            print(color.RED + "Early Stop!")

        self._save_models(gen_fname="generator_final", dscr_fname="discriminator_final")
        self._save_learning_data()

    # want to save when discriminator loss is low, but rewards are highest
    def _train_generator(self, epoch):
        # Train the generator for G_STEPS
        total_update = 0.0
        total_rewards = 0.0
        for gstep in range(const.G_STEPS):
            sdl = self.sdlf.get_subset_dataloader(const.BATCH_SIZE)
            data = next(iter(sdl))

            chord_roots, chord_types = data[const.CR_SEQS_KEY], data[const.CT_SEQS_KEY]
            seqs = data[const.SEQS_KEY]
            # Generate a batch of sequences
            # TODO: switch out the constants for params from cr and ct
            samples = self.generator.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN, chord_roots, chord_types,
                                            seed=seqs[:, :48])
            samples, chord_roots, chord_types = self.prepare_vars(samples, chord_roots, chord_types)

            # Calculate MLE rewards using the saved pretrained generator
            preds = self.generator.forward(samples, chord_roots, chord_types)
            mle_preds = self.reward_rnn.forward(samples, chord_roots, chord_types)
            kl_div = torch.sum(torch.exp(mle_preds) * (mle_preds - preds), dim=-1)
            mle_rewards = 1 - kl_div 
            mle_rewards[mle_rewards < 0] = 0
            mle_rewards[mle_rewards > 1] = 1
            # Calculate rewards for generator state using monte carlo rollout
            # rewards are equal to the average (over all rollouts) of the discriminator's likelihood of realism score
            rewards = self.rollout.get_rewards(samples, chord_roots, chord_types, const.NUM_ROLLOUTS, self.discriminator.cpu())
            # Because the discriminator gives results in negative log likelihood, exponent it
            rewards = torch.exp(torch.Tensor(rewards)).contiguous().view((-1,))
            rewards = self.prepare_vars(rewards)
            total_rewards += torch.mean(rewards)
            # set the discriminator back to cuda (might not be necessary)
            if self.args.cuda and torch.cuda.is_available():
                self.discriminator = self.discriminator.cuda()

            print(color.YELLOW + '::INFO:: Mean Episode Dscr Rewards: %.5f' % torch.mean(rewards))
            print(color.CYAN + '::INFO:: Mean Episode MLE Rewards: %.5f' % torch.mean(mle_rewards))

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
            policy_probs, adv_update = self.policy_update(prob, targets, rewards, mle_rewards) 
            adv_update = adv_update / const.BATCH_SIZE  # from suragnair/seqGAN
            print(color.GREEN + '::INFO:: Mean Certainty (p(a|s)): %.5f' % torch.exp(torch.mean(policy_probs)))
            print(color.CYAN + 'Adv Epoch [%d], Gen Step [%d] - Policy Update: %f' % (epoch, gstep, adv_update))
            total_update += adv_update
            self.policy_update.zero_grad()
            adv_update.backward()
            self.policy_optimizer.step()

            # Update the parameters of the generator copy inside the rollout object.
            self.rollout.update_params()

            # Check how our MLE validation changes with GAN loss. We've noticed it going up, but are unsure
            # if this is a good metric by which to validate for this type of training.
            valid_loss = self._eval_epoch(self.generator, self.valid_iter, self.eval_loss)
            print(color.CYAN + 'Adv Epoch [%d], Gen Step [%d] - Valid Loss: %f' % (epoch, gstep, valid_loss))

        return total_update / const.G_STEPS, total_rewards / const.G_STEPS

    def _train_discriminator(self, epoch):
        total_dscr_loss = 0.0
        for data_gen in range(const.D_DATA_GENS):
            data_iter = self._get_dscr_data_iter(const.NUM_TRAIN_SAMPLES)
            for dstep in range(const.D_STEPS):
                loss = self._train_dscr_epoch(data_iter) 
                total_dscr_loss += loss
                print(color.CYAN + 'Adv Epoch [%d], Dscr Gen [%d], Dscr Step [%d] - Loss: %f' % (epoch, data_gen, dstep, loss))
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

    def _save_models(self, gen_fname="generator", dscr_fname="discriminator"):
        """
        Save the model and parameters needed to reinstantiate it at the end of training 
        """
        print(color.CYAN + "::INFO:: Saving models ... ")
        gen_model_inputs = {'vocab_size': const.VOCAB_SIZE,
                            'embed_dim': const.GEN_EMBED_DIM,
                            'hidden_dim': const.GEN_HIDDEN_DIM,
                            'use_cuda': self.args.cuda}

        torch.save({'state_dict': self.generator.state_dict(),
                    'model_inputs': gen_model_inputs}, op.join(self.run_dir, gen_fname + '.pt'))

        dscr_model_inputs = {'vocab_size': const.VOCAB_SIZE, 
                             'embed_dim': const.DSCR_EMBED_DIM,
                             'filter_sizes': const.DSCR_FILTER_LENGTHS,
                             'num_filters': const.DSCR_NUM_FILTERS,
                             'output_dim': const.DSCR_NUM_CLASSES,
                             'dropout': const.DSCR_DROPOUT}

        torch.save({'state_dict': self.discriminator.state_dict(),
                    'model_inputs': dscr_model_inputs}, op.join(self.run_dir, dscr_fname + '.pt'))

    def _save_learning_data(self):
        print(color.CYAN + "::INFO:: Saving losses, rewards, updates ... ")
        torch.save({'gen_updates': self.gen_updates, 
                    'rewards': self.avg_rewards,
                    'dscr_losses': self.dscr_losses,
                    'losses': self.losses}, op.join(self.run_dir, 'learning_data.pt'))

