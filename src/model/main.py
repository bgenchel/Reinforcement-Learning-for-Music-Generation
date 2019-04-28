"""
adapted from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import argparse
import bdb
import numpy as np
import random
import os
import os.path as op
import traceback
import torch
from pathlib import Path
# local
from generator import Generator
from discriminator import Discriminator as Discriminator
from utils import constants as const
from utils.data.datasets import HDF5Dataset
from utils.data.dataloaders import SplitDataLoader
from utils.training import GeneratorPretrainer, DiscriminatorPretrainer, AdversarialRLTrainer

parser = argparse.ArgumentParser(description="Training Parameter")
parser.add_argument('-glr', '--gen_learning_rate', default=1e-3, type=float, help="learning rate for generator")
parser.add_argument('-aglr', '--adv_gen_learning_rate', default=1e-3, type=float, 
                    help="learning rate for generator during adversarial training")
parser.add_argument('-dlr', '--dscr_learning_rate', default=1e-3, type=float, help="learning rate for discriminator")
parser.add_argument('-adlr', '--adv_dscr_learning_rate', default=1e-4, type=float, 
                    help=" adversarial learning rate for discriminator")
parser.add_argument('-fpt', '--force_pretrain', default=False, action='store_true', 
                    help="force pretraining of both generator and discriminator, instead of loading from cache.")
parser.add_argument('-fpg', '--force_pretrain_gen', default=False, action='store_true', 
                    help="force pretraining of generator, instead of loading from cache.")
parser.add_argument('-fpd', '--force_pretrain_dscr', default=False, action='store_true', 
                    help="force pretraining of discriminator, instead of loading from cache.")
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


def main(pretrain_dataset, rl_dataset, args):
    ##############################################################################
    # Setup
    ##############################################################################
    # set random seeds
    random.seed(const.SEED)
    np.random.seed(const.SEED)

    # load datasets
    pt_train_loader, pt_valid_loader = SplitDataLoader(pretrain_dataset, batch_size=const.BATCH_SIZE, drop_last=True).split()

    # Define Networks
    generator = Generator(const.VOCAB_SIZE, const.GEN_EMBED_DIM, const.GEN_HIDDEN_DIM, device, args.cuda)
    discriminator = Discriminator(const.VOCAB_SIZE, const.DSCR_EMBED_DIM, const.DSCR_FILTER_LENGTHS, 
                                  const.DSCR_NUM_FILTERS, const.DSCR_NUM_CLASSES, const.DSCR_DROPOUT)

    # if torch.cuda.device_count() > 1:
        # print("Using", torch.cuda.device_count(), "GPUs.")
        # generator = nn.DataParallel(generator)
        # discriminator = nn.DataParallel(discriminator)
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
    print('#' * 80)
    print('Generator Pretraining')
    print('#' * 80)
    if (not (args.force_pretrain or args.force_pretrain_gen)) and op.exists(GEN_MODEL_CACHE):
        print('Loading Pretrained Generator ...')
        checkpoint = torch.load(GEN_MODEL_CACHE)
        generator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained for %d epochs.' % checkpoint['epochs'])
        print('::INFO:: Final Training Loss - %.5f' % checkpoint['train_loss'])
        print('::INFO:: Final Validation Loss - %.5f' % checkpoint['valid_loss'])
    else:
        try:
            print('Pretraining Generator with MLE ...')
            GeneratorPretrainer(generator, pt_train_loader, pt_valid_loader, PT_CACHE_DIR, device, args).train()
        except KeyboardInterrupt:
            print('Stopped Generator Pretraining Early.')

    # Pretrain Discriminator on real data and data from the pretrained generator. If a pretrained Discriminator
    # already exists, load it and display its stats
    print('#' * 80)
    print('Discriminator Pretraining')
    print('#' * 80)
    if (not (args.force_pretrain or args.force_pretrain_dscr)) and op.exists(DSCR_MODEL_CACHE):
        print("Loading Pretrained Discriminator ...")
        checkpoint = torch.load(DSCR_MODEL_CACHE)
        discriminator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained on %d data generations.' % checkpoint['data_gens'])
        print('::INFO:: Model was trained for %d epochs per data generation.' % checkpoint['epochs_per_gen'])
        print('::INFO:: Final Loss - %.5f' % checkpoint['loss'])
    else:
        print('Pretraining Discriminator ...')
        try:
            DiscriminatorPretrainer(discriminator, rl_dataset, PT_CACHE_DIR, TEMP_DATA_DIR, device, args).train(generator)
        except KeyboardInterrupt:
            print('Stopped Discriminator Pretraining Early.')
    ##############################################################################

    ##############################################################################
    # Adversarial Training 
    ##############################################################################
    print('#' * 80)
    print('Adversarial Training')
    print('#' * 80)
    AdversarialRLTrainer(generator, discriminator, rl_dataset, TEMP_DATA_DIR, pt_valid_loader, device, args).train()
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
