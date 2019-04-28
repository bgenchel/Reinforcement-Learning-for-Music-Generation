"""
A script used to generate midi files from the trained generator. 
TODO: allow specification of run dir as a command line argument
"""
import h5py
import json
import os
import os.path as op
import numpy as np
import random
import torch
from generator import Generator
from pathlib import Path
import pdb

import utils.constants as const
from utils.reverse_pianoroll import piano_roll_to_pretty_midi

A0 = 21
B0 = 23

NUM_GENS = 10
SEQ_LEN = 384
SEED_LEN = 48  # 2 beats

RUN_LABEL = 'Keep-Apr28-19_03:32-MP_Final_Train'

CT_DICT = {0: {"label": "n.c.",
               "components": {}},
           1: {"label": "maj",
               "components": {3: 0, 5: 0}},
           2: {"label": "maj7",
               "components": {3: 0, 5: 0, 7: 0}},
           3: {"label": "m",
               "components": {3: -1, 5: 0}},
           4: {"label": "m7",
               "components": {3: -1, 5: 0, 7: -1}},
           5: {"label": "5",
               "components": {5: 0}},
           6: {"label": "aug",
               "components": {3: 0, 5: 1}},
           7: {"label": "aug7",
               "components": {3: 0, 5: 1, 7: -1}},
           8: {"label": "aug9",
               "components": {3: 0, 5: 1, 7: -1, 9: 0}},
           9: {"label": "dim",
               "components": {3: -1, 5: -1}},
           10: {"label": "m7b5",
                "components": {3: -1, 5: -1, 7: -1}},
           11: {"label": "dim7",
                "components": {3: -1, 5: -1, 7: -2}},
           12: {"label": "7",
                "components": {3: 0, 5: 0, 7: -1}},
           13: {"label": "m(maj7)",
                "components": {3: -1, 5: 0, 7: 0}},
           14: {"label": "6",
                "components": {3: 0, 5: 0, 6: 0}},
           15: {"label": "m6",
                "components": {3: -1, 5: 0, 6: 0}},
           16: {"label": "9",
                "components": {3: 0, 5: 0, 7: -1, 9: 0}},
           17: {"label": "maj9",
                "components": {3: 0, 5: 0, 7: 0, 9: 0}},
           18: {"label": "m9",
                "components": {3: -1, 5: 0, 7: -1, 9: 0}},
           19: {"label": "maj69",
                "components": {3: 0, 5: 0, 6: 0, 9: 0}},
           20: {"label": "11",
                "components": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0}},
           21: {"label": "maj11",
                "components": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0}},
           22: {"label": "m11",
                "components": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0}},
           23: {"label": "13",
                "components": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0}},
           24: {"label": "maj13",
                "components": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0, 13: 0}},
           25: {"label": "m13",
                "components": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0}},
           26: {"label": "sus2",
                "components": {2: 0, 5: 0}},
           27: {"label": "sus4",
                "components": {4: 0, 5: 0}}}

DEGREE_MAP = {3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 9: 14, 11: 17, 13: 21}


def main():
    device = torch.device("cpu")
    # load saved model files
    rl_model_dir = op.join(os.getcwd(), 'runs', 'charlie_parker', RUN_LABEL)
    model_dict = torch.load(op.join(rl_model_dir, 'generator_final.pt'), map_location='cpu')
    rl_model_inputs = model_dict['model_inputs']
    rl_model_inputs['use_cuda'] = False
    rl_model_inputs['device'] = device
    # reconstitute models
    rl_gen = Generator(**rl_model_inputs)
    rl_gen.load_state_dict(model_dict['state_dict'])
    rl_gen.eval()
    # output
    rl_outdir = op.join(os.getcwd(), 'generated', 'charlie_parker', RUN_LABEL, 'fully_trained')
    if not op.exists(rl_outdir):
        os.makedirs(rl_outdir)

    # load saved model files
    mle_model_dir = op.join(os.getcwd(), 'pretrained', 'charlie_parker')
    mle_model_state = torch.load(op.join(mle_model_dir, 'generator.pt'), map_location='cpu')['state_dict']
    # reconstitute models
    mle_gen = Generator(89, 8, 128, device, False)
    mle_gen.load_state_dict(mle_model_state)
    mle_gen.eval()
    # output
    mle_outdir = op.join(os.getcwd(), 'generated', 'charlie_parker', RUN_LABEL, 'pretrained')
    if not op.exists(mle_outdir):
        os.makedirs(mle_outdir)

    data_dir = op.join(str(Path(op.abspath(__file__)).parents[2]), 'data', 'processed', 'charlie_parker-hdf5')
    dataset = h5py.File(op.join(data_dir, 'charlie_parker-dataset-rl.h5'), 'r')

    for i in range(NUM_GENS):
        index = random.choice(range(len(dataset[const.SEQS_KEY])))
        base_seq = dataset[const.SEQS_KEY][index]
        chord_roots = dataset[const.CR_SEQS_KEY][index]
        chord_types = dataset[const.CT_SEQS_KEY][index]

        seed = torch.LongTensor(base_seq[:SEED_LEN]).unsqueeze(0)
        cr_inpt = torch.LongTensor(chord_roots).unsqueeze(0)
        ct_inpt = torch.LongTensor(chord_types).unsqueeze(0)
        # fully trained
        samples = rl_gen.sample(1, SEQ_LEN, cr_inpt, ct_inpt, seed=seed)
        sequence_to_midi(op.join(rl_outdir, '%d.mid' % i), samples.squeeze(0))
        harmony_to_midi(op.join(rl_outdir, '%d_harmony.mid' % i), chord_roots, chord_types)
        # pre trained
        samples = mle_gen.sample(1, SEQ_LEN, cr_inpt, ct_inpt, seed=seed)
        sequence_to_midi(op.join(mle_outdir, '%d.mid' % i), samples.squeeze(0))
        harmony_to_midi(op.join(mle_outdir, '%d_harmony.mid' % i), chord_roots, chord_types)

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


def harmony_to_midi(path, chord_roots, chord_types):
    pr = np.zeros([128, len(chord_roots)])
    for i, (cr, ct) in enumerate(zip(chord_roots, chord_types)):
        if cr == 0 or ct == 0:  # N.C.
            continue
        else:
            root_pos = B0 + cr + 12
            pr[root_pos, i] = 1
            try:
                for k, v in CT_DICT[ct]['components'].items():
                    pr[root_pos + 12 + DEGREE_MAP[k] + v, i] = 1
            except TypeError:
                pdb.set_trace()

    # Convert piano roll into MIDI file
    pm = piano_roll_to_pretty_midi(pr, fs=100)
    pm.write(path)


if __name__ == "__main__":
    main()
