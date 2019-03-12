"""
A script used to generate midi files from the trained generator. 
TODO: allow specification of run dir as a command line argument
"""
import json
import os
import os.path as op
import numpy as np
import pathlib
import src
import torch

src.path.append(str(Path(__file__).parents[2]))
from utils.reverse_pianoroll import piano_roll_to_pretty_midi

def main():
    # load saved model files
    model_dir = op.join('runs', 'Nov27-18_14:16:33')
    model_inputs = json.load(open(op.join(model_dir, 'model_inputs.json'), 'r'))
    model_state = torch.load(op.join(model_dir, 'generator_state.pt'), map_location='cpu')
    # reconstitute models
    gen = Generator(**model_inputs)
    gen.load_state_dict(model_state)
    gen.eval()


def sequence_to_midi(path, sequence):
    # Convert tokens into a piano roll
    pr = np.zeros([128, len(sequence)])
    for i in range(len(sequence)):
        pr[sequence[i], i] = 1

    # Convert piano roll into MIDI file
    pm = piano_roll_to_pretty_midi(pr, fs=1 / 0.4)
    pm.write(path)


if __name__ == "__main__":
    main()
