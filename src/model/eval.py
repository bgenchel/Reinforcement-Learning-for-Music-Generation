import pdb
import h5py
import os.path as op
import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from utils import constants as const
from utils import helpers as hlp
from utils.data.datasets import HDF5Dataset
from make_music import sequence_to_midi
from generator import Generator

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from src.evaluation.bleu import BleuScore

VOCAB_SIZE = 89
EMBED_DIM = 8
HIDDEN_DIM = 128
SEQ_LEN = 384
BATCH_SIZE = 64
SEED_LEN = 48

RUN_LABEL = 'Keep-Apr28-19_03:32-MP_Final_Train'

torch.cuda.set_device(1)
DEVICE = torch.device("cuda:1")


def listify(tensor):
    return tensor.cpu().numpy().tolist()


def bleu():
    """
    Used to calculate the BLEU score across the pretrained generator and adversarially trained generator.
    :return:
    """
    pt_preds, ft_preds, targets = get_predictions()

    print("Calculating BLEU Scores")
    bs = BleuScore(const.GEN_SEQ_LEN)
    pt_bleu = bs.evaluate_bleu_score(pt_preds, targets)
    ft_bleu = bs.evaluate_bleu_score(ft_preds, targets)

    print("BLEU Score for pretrained generator: {}".format(pt_bleu))
    print("BLEU Score for fully_trained generator: {}".format(ft_bleu))


def render_midi(num_seqs):
    """
    Renders the first num_seqs of the sequences to individual MIDI files.
    :return:
    """
    pt_preds, ft_preds, targets = get_predictions()

    reference_path = "eval_reference"
    pretrained_path = "eval_pretrained"
    fully_trained_path = "eval_fully_trained"

    if not op.exists(reference_path):
        os.mkdir(reference_path)
    if not op.exists(pretrained_path):
        os.mkdir(pretrained_path)
    if not op.exists(fully_trained_path):
        os.mkdir(fully_trained_path)

    for i in tqdm(range(num_seqs)):
        sequence_to_midi(op.join(reference_path, str(i) + "_reference.mid"), targets[i])
        sequence_to_midi(op.join(pretrained_path, str(i) + "_pretrained.mid"), pt_preds[i])
        sequence_to_midi(op.join(fully_trained_path, str(i) + "_fully_trained.mid"), ft_preds[i])


def get_predictions():
    """
    Loads the Nottingham dataset, returns the target sequences, generations from the pretrained generator, and
    generations from the adversarially trained generator.
    :return:
    """
    print("Loading Data ... ")
    data_dir = op.join(str(Path(op.abspath(__file__)).parents[2]), 'data', 'processed', 'charlie_parker-hdf5')
    dataset = HDF5Dataset(op.join(data_dir, "charlie_parker-dataset-mle.h5")) 
    dataloader = DataLoader(dataset, batch_size=const.BATCH_SIZE, drop_last=True, shuffle=True)

    pretrained = Generator(const.VOCAB_SIZE, const.GEN_EMBED_DIM, const.GEN_HIDDEN_DIM, DEVICE, use_cuda=True).cuda()
    pt_state = torch.load(op.join('pretrained', 'charlie_parker', 'generator.pt'))['state_dict']
    pretrained.load_state_dict(pt_state)

    fully_trained = Generator(const.VOCAB_SIZE, const.GEN_EMBED_DIM, const.GEN_HIDDEN_DIM, DEVICE, use_cuda=True).cuda()
    ft_state = torch.load(op.join('runs', 'charlie_parker', RUN_LABEL, 'generator.pt'))['state_dict']
    fully_trained.load_state_dict(ft_state)

    pt_preds = []
    ft_preds = []
    targets = []
    print("Generating Predictions ... ")
    # pdb.set_trace()
    for data in tqdm(dataloader):
        seqs, cr_seqs, ct_seqs = data[const.SEQS_KEY], data[const.CR_SEQS_KEY], data[const.CT_SEQS_KEY]
        target_seqs = data[const.TARGETS_KEY]

        seqs, cr_seqs, ct_seqs, target_seqs = hlp.prepare_vars(True, DEVICE, seqs, cr_seqs, ct_seqs, target_seqs)

        pt_pred = pretrained.forward(seqs, cr_seqs, ct_seqs).argmax(2)
        ft_pred = fully_trained.forward(seqs, cr_seqs, ct_seqs).argmax(2)
        pt_preds.extend(listify(pt_pred))
        ft_preds.extend(listify(ft_pred))
        targets.extend(listify(target_seqs))

    return targets, pt_preds, ft_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render_midi', action='store_true')
    parser.add_argument('--num_midi_samples', default=1000)
    parser.add_argument('-b', '--compute_bleu', action='store_true')
    args = parser.parse_args()

    if args.render_midi:
        render_midi(int(args.num_midi_samples))

    if args.compute_bleu:
        bleu()
