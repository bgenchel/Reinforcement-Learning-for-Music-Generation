"""
adapted from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import argparse
import bdb
import copy
import json
import math
import numpy as np
import random
import os
import os.path as op
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from gan_loss import GANLoss
from data_iter import GenDataset, DscrDataset

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from utils import constants as const
from utils.data.datasets import HDF5Dataset
from utils.data.dataloaders import SplitDataLoader

import pdb

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

args.cuda = False
if torch.cuda.is_available() and (not args.no_cuda):
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda:%d" % args.cuda_device)
    args.cuda = True
else:
    torch.set_device("cpu")
    device = torch.device("cpu")

# Paths
ROOT_DIR = str(Path(op.abspath(__file__)).parents[2])
DATA_DIR = op.join(ROOT_DIR, "data", "processed", "%s-hdf5" % args.dataset)
PT_CACHE_DIR = op.join(os.getcwd(), "pretrained", args.dataset)
if not op.exists(PT_CACHE_DIR):
    os.makedirs(PT_CACHE_DIR)
TEMP_DATA_DIR = op.join(os.getcwd(), "temp_data", args.dataset)
if not op.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)

gen_model_cache = op.join(PT_CACHE_DIR, 'generator.pt')
dscr_model_cache = op.join(PT_CACHE_DIR, 'discriminator.pt')

# General Training Paramters
SEED = 88 # for the randomize, no reason behind this number
BATCH_SIZE = 64
GAN_TRAIN_EPOCHS = 200 # number of adversarial training epochs
# NUM_SAMPLES = 5000 # num samples in each data file for training discriminator
NUM_SAMPLES = 500 # num samples in each data file for training discriminator
VOCAB_SIZE = 89 # 1 tick has how many dims?

# Pretraining Params
# GEN_PRETRAIN_EPOCHS = 120
GEN_PRETRAIN_EPOCHS = 50
DSCR_PRETRAIN_DATA_GENS = 5
DSCR_PRETRAIN_EPOCHS = 3

# Adversarial Training Params
NUM_ROLLOUTS = 16
# NUM_ROLLOUTS = 1
G_STEPS = 1
D_DATA_GENS = 4
D_STEPS = 2
# D_DATA_GENS = 1
# D_STEPS = 1

# Generator Model Params
gen_embed_dim = 64
gen_hidden_dim = 128 # originally 64
gen_seq_len = const.TICKS_PER_MEASURE * 4 # generate 4 bars

# Discriminator Model Parameters
dscr_embed_dim = 128
dscr_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dscr_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dscr_dropout = 0.75
dscr_num_classes = 2 

######
# This method allows us to train the model stochastically, as opposed to training over the full dataset, which
# for nottingham is over 170K samples. Each time we need a new real.data file, we first get
# a new dataloader using this method, that has a new set of NUM_SAMPLES random samples from the original dataset.
def get_subset_dataloader(dataset):
    print('entered get_subset_dataloader')
    indices = random.sample(range(len(dataset)), min(len(dataset), NUM_SAMPLES))
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(indices), drop_last=True)


# Generates `num_batches` batches of size BATCH_SIZE from the generator. Stores the data in `output_file`
def create_generated_data_file(model, num_batches, outpath):
    print('creating generated data file ...')
    samples = []
    for _ in range(num_batches):
        sample_batch = model.sample(BATCH_SIZE, gen_seq_len).cpu().data.numpy().tolist()
        samples.extend(sample_batch)
    
    if not op.exists(op.dirname(outpath)):
        os.makedirs(op.dirname(outpath))
    with open(outpath, 'w') as fout:
        for sample in samples:
            str_sample = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % str_sample)
    return


# Iterates through `data_iter` and stores all its targets in `output_file`.
def create_real_data_file(data_iter, output_file):
    print('creating real data file ...')
    samples = []
    # data_iter = iter(data_iter)
    for data in tqdm(data_iter, desc=' - Create Real Data File', leave=False):
        sample_batch = list(data[const.SEQS_KEY].numpy())
        samples.extend(sample_batch)

    with open(output_file, 'w') as fout:
        for sample in samples:
            str_sample = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % str_sample)
    return


# trains `model` for one epoch using data from `data_iter`. train_type refers to the option of training the model
# using either 'full sequence' or 'next step' (teacher forcing). For this model, we only use full sequence training
# because it is how it was done in the original.
def pretrain_epoch(model, data_iter, loss_fn, optimizer, pretrain_gen=False):
    total_loss = 0.0
    total_words = 0.0
    total_batches = 0.0
    for data in tqdm(data_iter, desc=' - Training', leave=False):
        data_var = data["sequences"]
        target_var = data['targets']

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
        ############################
        # just temporary, for debugging
        # if total_batches > 100:
            # break
        ###########################

    if type(model) == Discriminator:
        return total_loss / (total_batches * BATCH_SIZE)
    else:
        return total_loss / total_words

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
        return total_loss / (total_batches * BATCH_SIZE)
    else:
        return total_loss / total_words

def preeval_epoch(model, data_iter, loss_fn):
    total_loss = 0.0
    total_words = 0.0
    count = 0 # temp for debugging
    with torch.no_grad():
        for data in tqdm(data_iter, desc= " - Evaluation", leave=False):
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

            ############################
            # just temporary, for debugging
            count += 1
            if count > 100:
                break
            ###########################

    if type(model) == Discriminator:
        return total_loss / total_batches
    else:
        return total_loss / total_words

# evaluate `model`'s performance on data from `data_iter`. See above for train_type.
def eval_epoch(model, data_iter, loss_fn):
    total_loss = 0.0
    total_words = 0.0
    with torch.no_grad():
        for data in tqdm(data_iter, desc= " - Evaluation", leave=False):
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

    if type(model) == Discriminator:
        return total_loss / total_batches
    else:
        return total_loss / total_words


def main(pretrain_dataset, rl_dataset, args):
    ##############################################################################
    # Setup
    ##############################################################################
    gen_data_path = op.join(TEMP_DATA_DIR, "generated.data")
    real_data_path = op.join(TEMP_DATA_DIR, "real.data")
    gen_model_cache = op.join(PT_CACHE_DIR, "generator.pt")
    dscr_model_cache = op.join(PT_CACHE_DIR, "discriminator.pt")

    # for keeping track of loss curves so we can plot them later
    pt_gen_train_loss = []
    pt_gen_valid_loss = []
    pt_dscr_loss = [] 
    adv_gen_loss = []
    adv_dscr_loss = []

    # set random seeds
    random.seed(SEED)
    np.random.seed(SEED)

    # load datasets
    pt_train_loader, pt_valid_loader = SplitDataLoader(pretrain_dataset, batch_size=BATCH_SIZE, drop_last=True).split()

    # Define Networks
    generator = Generator(VOCAB_SIZE, gen_embed_dim, gen_hidden_dim, args.cuda)
    discriminator = Discriminator(VOCAB_SIZE, dscr_embed_dim, dscr_filter_sizes, dscr_num_filters, dscr_num_classes, dscr_dropout)
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
    if not args.force_pretrain and op.exists(gen_model_cache):
        print('Loading Pretrained Generator ...')
        checkpoint = torch.load(gen_model_cache)
        generator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained for %d epochs.' % checkpoint['epochs'])
        print('::INFO:: Final Training Loss - %.5f' % checkpoint['train_loss'])
        print('::INFO:: Final Validation Loss - %.5f' % checkpoint['valid_loss'])
    else:
        print('Pretraining Generator with MLE ...')
        gen_criterion = nn.NLLLoss(size_average=False)
        gen_optimizer = optim.Adam(generator.parameters(), lr=args.gen_learning_rate)
        min_valid_loss = float('inf')

        if args.cuda and torch.cuda.is_available():
            gen_criterion = gen_criterion.cuda()

        for epoch in range(GEN_PRETRAIN_EPOCHS):
            train_loss = pretrain_epoch(generator, pt_train_loader, gen_criterion, gen_optimizer)
            pt_gen_train_loss.append(train_loss)
            print('::PRETRAIN GEN:: Epoch [%d] Training Loss: %f'% (epoch, train_loss))

            valid_loss = preeval_epoch(generator, pt_valid_loader, gen_criterion)
            pt_gen_valid_loss.append(valid_loss)
            print('::PRETRAIN GEN:: Epoch [%d] Validation Loss: %f'% (epoch, valid_loss))

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                print('Caching Pretrained Generator ...')
                torch.save({'state_dict': generator.state_dict(),
                            'epochs': epoch + 1,
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                            'datetime': datetime.now().isoformat()}, gen_model_cache)

        torch.save({'train_losses': pt_gen_train_loss,
                    'valid_losses': pt_gen_valid_loss}, op.join(PT_CACHE_DIR, 'generator_losses.pt'))

    # Pretrain Discriminator on real data and data from the pretrained generator. If a pretrained Discriminator
    # already exists, load it and display its stats
    if not args.force_pretrain and op.exists(dscr_model_cache):
        print("Loading Pretrained Discriminator ...")
        checkpoint = torch.load(dscr_model_cache)
        discriminator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained on %d data generations.' % checkpoint['data_gens'])
        print('::INFO:: Model was trained for %d epochs per data generation.' % checkpoint['epochs'])
        print('::INFO:: Final Loss - %.5f' % checkpoint['loss'])
    else:
        print('Pretraining Discriminator ...')
        dscr_criterion = nn.NLLLoss(size_average=False)
        dscr_optimizer = optim.Adam(discriminator.parameters(), lr=args.dscr_learning_rate)
        if args.cuda and torch.cuda.is_available():
            dscr_criterion = dscr_criterion.cuda()
        for i in range(DSCR_PRETRAIN_DATA_GENS):
            # print('Creating real and fake datafiles ...')
            rl_data_loader = get_subset_dataloader(rl_dataset)
            create_generated_data_file(generator, len(rl_data_loader), gen_data_path)
            create_real_data_file(rl_data_loader, real_data_path)
            dscr_data_iter = DataLoader(DscrDataset(real_data_path, gen_data_path), batch_size=BATCH_SIZE, shuffle=True)
            for j in range(DSCR_PRETRAIN_EPOCHS):
                loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer)
                pt_dscr_loss.append(loss)
                print("::PRETRAIN DSCR:: Data Gen [%d] Epoch [%d] Loss: %f" % (i, j, loss))

        print('Caching Pretrained Discriminator ...')
        torch.save({'state_dict': discriminator.state_dict(),
                    'data_gens': DSCR_PRETRAIN_DATA_GENS,
                    'epochs': DSCR_PRETRAIN_EPOCHS,
                    'loss': loss,
                    'datetime': datetime.now().isoformat()}, dscr_model_cache)

        torch.save({'losses': pt_dscr_loss}, op.join(PT_CACHE_DIR, 'discriminator_losses.pt'))
    ##############################################################################

    data_loader = get_subset_dataloader(rl_dataset)
    # create real data file if it doesn't yet exist
    if not op.exists(real_data_path):
        print('Creating real data file...')
        create_real_data_file(data_loader, real_data_path)

    # create generated data file if it doesn't yet exist
    if not op.exists(gen_data_path):
        print('Creating generated data file...')
        create_generated_data_file(generator, len(data_loader), gen_data_path)

    ##############################################################################
    # Adversarial Training 
    ##############################################################################

    print('#'*100)
    print('Start Adversarial Training...\n')
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
    for epoch in range(GAN_TRAIN_EPOCHS):
        print("#" * 30 + "\nAdversarial Epoch [%d]\n" % epoch + "#"*30)
        total_gen_loss = 0.0
        ## Train the generator for G_STEPS
        for gstep in range(G_STEPS):
            # Generate a batch of sequences
            samples = generator.sample(BATCH_SIZE, gen_seq_len)

            # Construct a corresponding input step for the sequences: 
            # add a timestep of zeros before samples and then delete its last column.
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if args.cuda and torch.cuda.is_available():
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data)

            # Calculate rewards for generator state using monte carlo rollout
            # rewards are equal to the average (over all rollouts) of the discriminator's likelihood of realism score
            rewards = rollout.get_reward(samples, NUM_ROLLOUTS, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            # Because the discriminator gives results in negative log likelihood, exponent it
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if args.cuda and torch.cuda.is_available():
                rewards = rewards.cuda()

            # Make a forward prediction of the next step given the inputs using the generator
            prob = generator.forward(inputs)
            # adv_loss = gen_gan_loss(prob, targets, rewards)
            adv_loss = gen_gan_loss(prob, targets, rewards) / BATCH_SIZE # from suragnair/seqGAN
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

        adv_gen_loss.append(total_gen_loss / G_STEPS)
        # Save the model and parameters needed to reinstantiate it at the end of training 
        if (total_gen_loss / G_STEPS) < min_loss:
            model_inputs = {'vocab_size': VOCAB_SIZE,
                            'embed_dim': gen_embed_dim,
                            'hidden_dim': gen_hidden_dim,
                            'use_cuda': args.cuda}

            json.dump(model_inputs, open(op.join(run_dir, 'model_inputs.json'), 'w'), indent=4)
            torch.save(generator.state_dict(), op.join(run_dir, 'generator_state.pt'))

            torch.save({'adv_gen_losses': adv_gen_loss, 'adv_dscr_losses': adv_dscr_loss}, op.join(run_dir, 'losses.pt'))
        
        # Train the Discriminator for `D_STEPS` each on `D_DATA_GENS` sets of generated and sampled real data
        total_dscr_loss = 0.0
        for data_gen in range(D_DATA_GENS):
            data_loader = get_subset_dataloader(rl_dataset)
            create_generated_data_file(generator, len(data_loader), gen_data_path)
            create_real_data_file(data_loader, gen_data_path)
            dscr_data_iter = DataLoader(DscrDataset(real_data_path, gen_data_path), batch_size=BATCH_SIZE, shuffle=True)
            for dstep in range(D_STEPS):
                loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer)
                total_dscr_loss += loss
                print('Adv Epoch [%d], Dscr Gen [%d], Dscr Step [%d] - Loss: %f' % (epoch, data_gen, dstep, loss))
        adv_dscr_loss.append(total_dscr_loss / (D_DATA_GENS * D_STEPS))

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
