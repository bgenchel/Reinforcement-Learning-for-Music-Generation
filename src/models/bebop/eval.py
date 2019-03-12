import os.path as op
import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

import pdb
sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from make_music import sequence_to_midi
from data.parsing.datasets import NottinghamDataset
from evaluation.bleu import BleuScore
from generator import Generator

VOCAB_SIZE = 89
EMBED_DIM = 64
HIDDEN_DIM = 128
SEQ_LEN = 32
BATCH_SIZE = 128

def listify(tensor):
    return tensor.cpu().numpy().tolist()

def bleu():
    """
    Used to calculate the BLEU score across the pretrained generator and adversarially trained generator.
    :return:
    """
    pt_preds, ft_preds, targets = get_predictions()

    print("Calculating BLEU Scores")
    bs = BleuScore(SEQ_LEN)
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
    dataset = NottinghamDataset('../../../data/raw/nottingham-midi', seq_len=SEQ_LEN, data_format="nums")
    dataloader = DataLoader(dataset, batch_size=128, drop_last=True, shuffle=True)
    pretrained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=True)
    pretrained.cuda()
    pretrained.load_state_dict(torch.load(op.join('pretrained', 'generator.pt'), map_location='cpu')['state_dict'])
    fully_trained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=True)

    # Change this to the desired adversarial run
    best_run = 'Nov28-18_15:40:28'
    fully_trained.load_state_dict(torch.load(op.join('runs', best_run, 'generator_state.pt'), map_location='cpu'))
    fully_trained.cuda()

    pt_preds = []
    ft_preds = []
    targets = []
    print("Generating Predictions ... ")
    for (data, target) in tqdm(dataloader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        pt_pred = pretrained.forward(data).argmax(2)
        ft_pred = fully_trained.forward(data).argmax(2)
        pt_preds.extend(listify(pt_pred))
        ft_preds.extend(listify(ft_pred))
        targets.extend(listify(target))

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

