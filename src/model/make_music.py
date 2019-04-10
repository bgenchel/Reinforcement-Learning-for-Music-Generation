"""
A script used to generate midi files from the trained generator. 
TODO: allow specification of run dir as a command line argument
"""
import h5py
import json
import os
import os.path as op
import numpy as np
import pathlib
import random
import sys
import torch
from generator import Generator
from pathlib import Path

import pdb

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from utils.reverse_pianoroll import piano_roll_to_pretty_midi

A0 = 21

NUM_GENS = 100
SEQ_LEN = 384
SEED_LEN = 48 # 2 beats

def main():
    # load saved model files
    rl_model_dir = op.join(os.getcwd(),'runs', 'charlie_parker', 'Apr04-19_19:07:07')
    rl_model_inputs = json.load(open(op.join(rl_model_dir, 'model_inputs.json'), 'r'))
    rl_model_inputs['use_cuda'] = False
    rl_model_state = torch.load(op.join(rl_model_dir, 'generator_state.pt'), map_location='cpu')
    # reconstitute models
    rl_gen = Generator(**rl_model_inputs)
    rl_gen.load_state_dict(rl_model_state)
    rl_gen.eval()
    # output
    rl_outdir = op.join(os.getcwd(), 'generated', 'charlie_parker', 'pt')
    if not op.exists(rl_outdir):
        os.makedirs(rl_outdir)

    # load saved model files
    mle_model_dir = op.join(os.getcwd(),'pretrained', 'charlie_parker')
    mle_model_state = torch.load(op.join(mle_model_dir, 'generator.pt'), map_location='cpu')['state_dict']
    # reconstitute models
    mle_gen = Generator(89, 64, 128, False)
    mle_gen.load_state_dict(mle_model_state)
    mle_gen.eval()
    # output
    mle_outdir = op.join(os.getcwd(), 'generated', 'charlie_parker', 'ft')
    if not op.exists(mle_outdir):
        os.makedirs(mle_outdir)

    data_dir = op.join(str(Path(op.abspath(__file__)).parents[2]), 'data', 'processed', 'charlie_parker-hdf5')
    dataset = h5py.File(op.join(data_dir, 'charlie_parker-dataset-rl.h5'), 'r')

    for i in range(NUM_GENS):
        base_seq = dataset['sequences'][random.choice(range(len(dataset['sequences'])))]
        seed = torch.LongTensor(base_seq[:SEED_LEN]).unsqueeze(0)
        # fully trained
        samples = rl_gen.sample(1, SEQ_LEN, seed=seed)
        sequence_to_midi(op.join(rl_outdir, '%d.mid' % i), samples.squeeze(0))
        # pre trained
        samples = mle_gen.sample(1, SEQ_LEN, seed=seed)
        sequence_to_midi(op.join(mle_outdir, '%d.mid' % i), samples.squeeze(0))

def sequence_to_midi(path, sequence):
    # Convert tokens into a piano roll
    pr = np.zeros([128, len(sequence)])
    for i in range(len(sequence)):
        if sequence[i] == 0:
            continue
        else:
            pr[sequence[i] + A0, i] = 1

    # Convert piano roll into MIDI file
    pm = piano_roll_to_pretty_midi(pr, fs=100)
    pm.write(path)


if __name__ == "__main__":
    main()
