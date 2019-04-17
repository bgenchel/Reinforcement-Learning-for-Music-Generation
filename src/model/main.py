"""
adapted from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import argparse
import bdb
import json
import numpy as np
import random
import os
import os.path as op
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch.autograd import Variable
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from gan_loss import GANLoss

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from utils import constants as const
from utils.data.datasets import HDF5Dataset
from utils.data.dataloaders import SplitDataLoader
from utils.training import GeneratorPretrainer, DiscriminatorTrainer

parser = argparse.ArgumentParser(description="Training Parameter")
parser.add_argument('-tt', '--train_type', choices=("full_sequence", "next_step"), 
                    default="full_sequence", help="how to train the network")
parser.add_argument('-glr', '--gen_learning_rate', default=1e-3, type=float, help="learning rate for generator")
parser.add_argument('-aglr', '--adv_gen_learning_rate', default=1e-3, type=float, 
                    help="learning rate for generator during adversarial training")
parser.add_argument('-dlr', '--dscr_learning_rate', default=1e-3, type=float, help="learning rate for discriminator")
parser.add_argument('-adlr', '--adv_dscr_learning_rate', default=1e-3, type=float, 
                    help=" adversarial learning rate for discriminator")
parser.add_argument('-fpt', '--force_pretrain', default=False, action='store_true', 
                    help="force pretraining of generator and discriminator, instead of loading from cache.")
parser.add_argument('-nc', '--no_cuda', action='store_true', help="don't use CUDA, even if it is available.")
parser.add_argument('-cd', '--cuda_device', default=0, type=int, help="Which GPU to use")
parser.add_argument('-d', '--dataset', choices=("charlie_parker", "bebop", "nottingham"), type=str, 
                    required=True, help="the dataset to train with.")
args = parser.parse_args()

if torch.cuda.is_available() and (not args.no_cuda):
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda:%d" % args.cuda_device)
    args.cuda = True
else:
    torch.set_device("cpu")
    device = torch.device("cpu")
    args.cuda = False

# Paths
ROOT_DIR = str(Path(op.abspath(__file__)).parents[2])
DATA_DIR = op.join(ROOT_DIR, "data", "processed", "%s-hdf5" % args.dataset)
PT_CACHE_DIR = op.join(os.getcwd(), "pretrained", args.dataset)
if not op.exists(PT_CACHE_DIR):
    os.makedirs(PT_CACHE_DIR)
TEMP_DATA_DIR = op.join(os.getcwd(), "temp_data", args.dataset)
if not op.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)

GEN_DATA_PATH = op.join(TEMP_DATA_DIR, "generated.data")
REAL_DATA_PATH = op.join(TEMP_DATA_DIR, "real.data")
GEN_MODEL_CACHE = op.join(PT_CACHE_DIR, "generator.pt")
DSCR_MODEL_CACHE = op.join(PT_CACHE_DIR, "discriminator.pt")


def train_epoch(model, data_iter, loss_fn, optimizer):
    total_loss = 0.0
    total_words = 0.0
    total_batches = 0.0
    for (data, target) in tqdm(data_iter, desc=' - Training', leave=False):
        data_var = Variable(data)
        target_var = Variable(target)

        if args.cuda and torch.cuda.is_available():
            data_var, target_var = data_var.cuda(), target_var.cuda()

        data_var = data_var.to(device)
        target_var = target_var.to(device)

        target_var = target_var.contiguous().view(-1)
        pred = model.forward(data_var)
        pred = pred.view(-1, pred.size()[-1])

        loss = loss_fn(pred, target_var)
        total_loss += loss.item()
        total_words += data_var.size(0) * data_var.size(1)
        total_batches += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if type(model) == Discriminator:
        return total_loss / (total_batches * const.BATCH_SIZE)
    else:
        return total_loss / total_words


# evaluate `model`'s performance on data from `data_iter`. See above for train_type.
def eval_epoch(model, data_iter, loss_fn):
    total_loss = 0.0
    total_words = 0.0
    total_batches = 0
    with torch.no_grad():
        for data in tqdm(data_iter, desc=" - Evaluation", leave=False):
            data_var = Variable(data["sequences"])
            target_var = Variable(data["targets"])

            if args.cuda and torch.cuda.is_available():
                data_var, target_var = data_var.cuda(), target_var.cuda()

            data_var = data_var.to(device)
            target_var = target_var.to(device)

            target_var = target_var.contiguous().view(-1)
            pred = model.forward(data_var)
            pred = pred.view(-1, pred.size()[-1])

            loss = loss_fn(pred, target_var)
            total_loss += loss.item()
            total_words += data_var.size(0) * data_var.size(1)
            total_batches += 1

    if type(model) == Discriminator:
        return total_loss / total_batches
    else:
        return total_loss / total_words


def main(pretrain_dataset, rl_dataset, args):
    ##############################################################################
    # Setup
    ##############################################################################
    # for keeping track of loss curves so we can plot them later
    adv_gen_loss = []
    adv_dscr_loss = []

    # set random seeds
    random.seed(const.SEED)
    np.random.seed(const.SEED)

    # load datasets
    pt_train_loader, pt_valid_loader = SplitDataLoader(pretrain_dataset, batch_size=const.BATCH_SIZE, drop_last=True).split()

    # Define Networks
    generator = Generator(const.VOCAB_SIZE, const.GEN_EMBED_DIM, const.GEN_HIDDEN_DIM, args.cuda)
    discriminator = Discriminator(const.VOCAB_SIZE, const.DSCR_EMBED_DIM, const.DSCR_FILTER_SIZES, 
                                  const.DSCR_NUM_FILTERS, const.DSCR_NUM_CLASSES, const.DSCR_DROPOUT)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    generator.to(device)
    discriminator.to(device)

    # set CUDA
    if args.cuda and torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    ##############################################################################

    ##############################################################################
    # Pre-Training
    ##############################################################################
    # Pretrain and save Generator using MLE, Load the Pretrained generator and display training stats 
    # if it already exists.
    if not args.force_pretrain and op.exists(op.join(PT_CACHE_DIR, 'generator.pt')):
        print('Loading Pretrained Generator ...')
        checkpoint = torch.load(GEN_MODEL_CACHE)
        generator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained for %d epochs.' % checkpoint['epochs'])
        print('::INFO:: Final Training Loss - %.5f' % checkpoint['train_loss'])
        print('::INFO:: Final Validation Loss - %.5f' % checkpoint['valid_loss'])
    else:
        print('Pretraining Generator with MLE ...')
        GeneratorPretrainer(generator, pt_train_loader, pt_valid_loader, PT_CACHE_DIR, device, args).train()

    # Pretrain Discriminator on real data and data from the pretrained generator. If a pretrained Discriminator
    # already exists, load it and display its stats
    if not args.force_pretrain and op.exists(DSCR_MODEL_CACHE):
        print("Loading Pretrained Discriminator ...")
        checkpoint = torch.load(DSCR_MODEL_CACHE)
        discriminator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained on %d data generations.' % checkpoint['data_gens'])
        print('::INFO:: Model was trained for %d epochs per data generation.' % checkpoint['epochs'])
        print('::INFO:: Final Loss - %.5f' % checkpoint['loss'])
    else:
        print('Pretraining Discriminator ...')
        DiscriminatorTrainer(discriminator, rl_dataset, PT_CACHE_DIR, TEMP_DATA_DIR, device, args).train(generator)
    ##############################################################################

    # data_loader = get_subset_dataloader(rl_dataset)
    # # create real data file if it doesn't yet exist
    # if not op.exists(REAL_DATA_PATH):
    #     print('Creating real data file...')
    #     create_real_data_file(data_loader, REAL_DATA_PATH)

    # # create generated data file if it doesn't yet exist
    # if not op.exists(GEN_DATA_PATH):
    #     print('Creating generated data file...')
    #     create_generated_data_file(generator, len(data_loader), GEN_DATA_PATH)

    ##############################################################################
    # Adversarial Training 
    ##############################################################################

    print('#'*100)
    print('Start Adversarial Training...\n')
    print('#'*100)
    # Instantiate the Rollout. Give it the generator so it can make its own internal copy of it 
    # which will only update `update_rate` of the full generator's update at each step
    rollout = Rollout(generator, update_rate=0.8, device=device)

    # Instantiate GANLoss and new optimizer for the generator with new learning rate.
    gen_gan_loss = GANLoss(use_cuda=args.cuda)
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=args.adv_gen_learning_rate)
    if args.cuda and torch.cuda.is_available():
        gen_gan_loss = gen_gan_loss.cuda()
        gen_criterion = gen_criterion.cuda()

    # Instantiate new loss and optimizer for the discriminator with new learning rate.
    dscr_criterion = nn.NLLLoss(size_average=False)
    dscr_optimizer = optim.Adam(discriminator.parameters(), lr=args.adv_dscr_learning_rate)
    if args.cuda and torch.cuda.is_available():
        dscr_criterion = dscr_criterion.cuda()

    run_dir = op.join("runs", args.dataset, datetime.now().strftime('%b%d-%y_%H:%M:%S'))
    if not op.exists(run_dir):
        os.makedirs(run_dir)

    # Train Adversarially for `GAN_TRAIN_EPOCHS` epochs
    min_loss = float('-inf')
    for epoch in range(const.GAN_TRAIN_EPOCHS):
        print("#" * 30 + "\nAdversarial Epoch [%d]\n" % epoch + "#"*30)
        total_gen_loss = 0.0
        # Train the generator for G_STEPS
        for gstep in range(const.G_STEPS):
            # Generate a batch of sequences
            samples = generator.sample(const.BATCH_SIZE, const.GEN_SEQ_LEN)

            # Construct a corresponding input step for the sequences: 
            # add a timestep of zeros before samples and then delete its last column.
            zeros = torch.zeros((const.BATCH_SIZE, 1)).type(torch.LongTensor)
            if args.cuda and torch.cuda.is_available():
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data)

            # Calculate rewards for generator state using monte carlo rollout
            # rewards are equal to the average (over all rollouts) of the discriminator's likelihood of realism score
            rewards = rollout.get_reward(samples, const.NUM_ROLLOUTS, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            # Because the discriminator gives results in negative log likelihood, exponent it
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if args.cuda and torch.cuda.is_available():
                rewards = rewards.cuda()

            # Make a forward prediction of the next step given the inputs using the generator
            prob = generator.forward(inputs)
            # adv_loss = gen_gan_loss(prob, targets, rewards)
            adv_loss = gen_gan_loss(prob, targets, rewards) / const.BATCH_SIZE  # from suragnair/seqGAN
            total_gen_loss += adv_loss
            print('Adv Epoch [%d], Gen Step [%d] - Train Loss: %f' % (epoch, gstep, adv_loss))
            # back propagate the GAN loss
            gen_gan_optm.zero_grad()
            adv_loss.backward()
            gen_gan_optm.step()

            # Check how our MLE validation changes with GAN loss. We've noticed it going up, but are unsure
            # if this is a good metric by which to validate for this type of training.
            valid_loss = eval_epoch(generator, pt_valid_loader, gen_criterion)
            print('Adv Epoch [%d], Gen Step [%d] - Valid Loss: %f' % (epoch, gstep, valid_loss))

            # Update the parameters of the generator copy inside the rollout object.
            rollout.update_params()

        adv_gen_loss.append(total_gen_loss / const.G_STEPS)
        # Save the model and parameters needed to reinstantiate it at the end of training 
        if (total_gen_loss / const.G_STEPS) < min_loss:
            model_inputs = {'vocab_size': const.VOCAB_SIZE,
                            'embed_dim': const.GEN_EMBED_DIM,
                            'hidden_dim': const.GEN_HIDDEN_DIM,
                            'use_cuda': args.cuda}

            json.dump(model_inputs, open(op.join(run_dir, 'model_inputs.json'), 'w'), indent=4)
            torch.save(generator.state_dict(), op.join(run_dir, 'generator_state.pt'))

            torch.save({'adv_gen_losses': adv_gen_loss, 'adv_dscr_losses': adv_dscr_loss}, op.join(run_dir, 'losses.pt'))
            min_loss = total_gen_loss / const.G_STEPS

        # Train the Discriminator for `D_STEPS` each on `D_DATA_GENS` sets of generated and sampled real data
        total_dscr_loss = 0.0
        for data_gen in range(const.D_DATA_GENS):
            data_loader = get_subset_dataloader(rl_dataset)
            create_generated_data_file(generator, len(data_loader), GEN_DATA_PATH)
            create_real_data_file(data_loader, GEN_DATA_PATH)
            dscr_data_iter = DataLoader(DscrDataset(REAL_DATA_PATH, GEN_DATA_PATH), batch_size=const.BATCH_SIZE, shuffle=True)
            for dstep in range(const.D_STEPS):
                loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer)
                total_dscr_loss += loss
                print('Adv Epoch [%d], Dscr Gen [%d], Dscr Step [%d] - Loss: %f' % (epoch, data_gen, dstep, loss))
        adv_dscr_loss.append(total_dscr_loss / (const.D_DATA_GENS * const.D_STEPS))
    ##############################################################################


if __name__ == '__main__':
    print("Loading Data ...")
    pretrain_dataset = HDF5Dataset(op.join(DATA_DIR, "%s-dataset-mle.h5" % args.dataset)) 
    rl_dataset = HDF5Dataset(op.join(DATA_DIR, "%s-dataset-rl.h5" % args.dataset)) 
    print("Done.")
    try:
        main(pretrain_dataset, rl_dataset, args)
    except (TypeError, KeyboardInterrupt, ValueError, OSError, RuntimeError, NameError, RuntimeError, bdb.BdbQuit) as e:
        print(traceback.format_exc())
        print("Closing Datasets ...")
        pretrain_dataset.terminate()
        rl_dataset.terminate()
