import argparse
import os
import os.path as op
import pickle as pkl
import pdb
import h5py
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
import utils.constants as const

ROOT_DIR = str(Path(op.abspath(__file__)).parents[2])
# default low and high for MIDI note range
A0 = 21 
C8 = 108

SEQ_LEN = 96 # 96 ticks to one measure
DTYPE = np.uint8


def prepare_mle_hdf5(file_path):
    print(file_path)
    hdf5 = h5py.File(file_path, 'w')
    # choosing int 8 because all these tokens should be less than 256
    hdf5.create_dataset(const.SEQS_KEY, shape=(0,) + (SEQ_LEN,), maxshape=(None,) + (SEQ_LEN,), dtype=DTYPE)
    hdf5.create_dataset(const.CR_SEQS_KEY, shape=(0,) + (SEQ_LEN,), maxshape=(None,) + (SEQ_LEN,), dtype=DTYPE)
    hdf5.create_dataset(const.CT_SEQS_KEY, shape=(0,) + (SEQ_LEN,), maxshape=(None,) + (SEQ_LEN,), dtype=DTYPE)
    hdf5.create_dataset(const.TARGETS_KEY, shape=(0,) + (SEQ_LEN,), maxshape=(None,) + (SEQ_LEN,), dtype=DTYPE)
    return hdf5

def prepare_rl_hdf5(file_path):
    print(file_path)
    hdf5 = h5py.File(file_path, 'w')
    # choosing int 8 because all these tokens should be less than 256
    hdf5.create_dataset(const.SEQS_KEY, shape=(0,) + (SEQ_LEN * 4,), maxshape=(None,) + (SEQ_LEN * 4,), dtype=DTYPE)
    hdf5.create_dataset(const.CR_SEQS_KEY, shape=(0,) + (SEQ_LEN * 4,), maxshape=(None,) + (SEQ_LEN * 4,), dtype=DTYPE)
    hdf5.create_dataset(const.CT_SEQS_KEY, shape=(0,) + (SEQ_LEN * 4,), maxshape=(None,) + (SEQ_LEN * 4,), dtype=DTYPE)
    return hdf5

def write_to_mle_hdf5(hdf5, seqs, targets, cr_seqs, ct_seqs):
    # Define references to the datasets
    h5_seqs = hdf5[const.SEQS_KEY]
    h5_targets = hdf5[const.TARGETS_KEY]
    h5_chord_root_seqs = hdf5[const.CR_SEQS_KEY]
    h5_chord_type_seqs = hdf5[const.CT_SEQS_KEY]
    # Resize the datasets
    add_num = len(seqs)
    curr_size, seq_len = h5_seqs.shape
    h5_seqs.resize(curr_size + add_num, axis=0)
    h5_targets.resize(curr_size + add_num, axis=0)
    h5_chord_root_seqs.resize(curr_size + add_num, axis=0)
    h5_chord_type_seqs.resize(curr_size + add_num, axis=0)
    # Add the data
    h5_seqs[curr_size:curr_size + add_num, :] = seqs
    h5_targets[curr_size:curr_size + add_num, :] = targets
    h5_chord_root_seqs[curr_size:curr_size + add_num, :] = cr_seqs
    h5_chord_type_seqs[curr_size:curr_size + add_num, :] = ct_seqs

def write_to_rl_hdf5(hdf5, seq, cr_seq, ct_seq):
    # Define references to the datasets
    h5_seqs = hdf5[const.SEQS_KEY]
    h5_chord_root_seqs = hdf5[const.CR_SEQS_KEY]
    h5_chord_type_seqs = hdf5[const.CT_SEQS_KEY]
    # Resize the datasets
    curr_size, seq_len = h5_seqs.shape
    h5_seqs.resize(curr_size + 1, axis=0)
    h5_chord_root_seqs.resize(curr_size + 1, axis=0)
    h5_chord_type_seqs.resize(curr_size + 1, axis=0)
    # Add the data
    h5_seqs[curr_size:curr_size + 1, :] = seq
    h5_chord_root_seqs[curr_size:curr_size + 1, :] = cr_seq
    h5_chord_type_seqs[curr_size:curr_size + 1, :] = ct_seq

def create_data_dict():
    return {const.TICK_KEY: [], const.CHORD_ROOT_KEY: [], const.CHORD_TYPE_KEY: []}

def get_seqs_and_targets(sequence):
    seqs, targets = [], []
    seq_len = SEQ_LEN
    if len(sequence.shape) == 1:
        padding = np.zeros((seq_len))
    else:
        padding = np.zeros((seq_len, sequence.shape[1]))
    sequence = np.concatenate((padding, sequence), axis=0)
    for i in range(sequence.shape[0] - seq_len):
        seqs.append(sequence[i:(i + seq_len)])
        targets.append(sequence[(i + 1):(i + seq_len + 1)])
    return seqs, targets

def main(args):
    parsed_dir = op.join(ROOT_DIR, "data", "interim", args.dataset + '-parsed') 
    out_dir = op.join(ROOT_DIR, "data", "processed", args.dataset + '-hdf5')
    if not op.exists(parsed_dir):
        raise Exception("{} does not exist -> no data to parse.".format(parsed_dir))
    if not op.exists(out_dir):
        os.makedirs(out_dir)

    h5_mle_fp = prepare_mle_hdf5(op.join(out_dir, args.dataset + "-dataset-mle.h5"))
    h5_rl_fp = prepare_rl_hdf5(op.join(out_dir, args.dataset + "-dataset-rl.h5"))
    
    file_count = 0
    for fname in tqdm(os.listdir(parsed_dir)):
        if op.splitext(fname)[1] != ".pkl":
            print("Skipping %s..." % fname)
            continue

        with open(op.join(parsed_dir, fname), "rb") as fp:
            song = pkl.load(open(op.join(parsed_dir, fname), "rb"))

        if song["metadata"]["time_signature"] != "4/4":
            print("Skipping %s because it isn't in 4/4." % fname)
            continue

        ########################################
        # Writing Pre-Training H5 File
        ########################################
        
        full_sequence = create_data_dict()
        for i, measure in enumerate(song["measures"]):
            for j, group in enumerate(measure["groups"]):
                chord_root = group[const.CHORD_KEY]['root']
                chord_type = group[const.CHORD_KEY]['type']
                # print('chord_root: {}, chord_type: {}'.format(chord_root, chord_type))
                full_sequence[const.CHORD_ROOT_KEY].extend([chord_root for _ in range(len(group['ticks']))])
                full_sequence[const.CHORD_TYPE_KEY].extend([chord_type for _ in range(len(group['ticks']))])

                formatted_ticks = []
                for tick in group['ticks']:
                    # formatted = tick[range_low - 1:range_high + 1]
                    if tick == 0 or tick < A0 or tick > C8:
                        formatted = 0
                    else:
                        formatted = tick - A0 + 1 # lower bound maps to 1, not 0 (rest)
                    formatted_ticks.append(formatted)

                full_sequence[const.TICK_KEY].extend(formatted_ticks)

        full_sequence = {k: np.array(v, dtype=DTYPE) for k, v in full_sequence.items()} 
        tick_seqs, tick_targets = get_seqs_and_targets(full_sequence[const.TICK_KEY])
        cr_seqs, cr_targets = get_seqs_and_targets(full_sequence[const.CHORD_ROOT_KEY])
        ct_seqs, ct_targets = get_seqs_and_targets(full_sequence[const.CHORD_TYPE_KEY])

        write_to_mle_hdf5(h5_mle_fp, tick_seqs, tick_targets, cr_seqs, ct_seqs)

        ########################################
        # Writing RL H5 File (4 measure phrases)
        ########################################

        # don't want to include pick up measures. Only complete, metric 4 bar phrases
        start_measure = 0
        while "rehearsal" not in song["measures"][start_measure].keys():
            start_measure += 1

            if start_measure == len(song["measures"]) - 1:
                print(song['metadata']['title'])

        while start_measure + 4 < len(song["measures"]):
            four_bar_phrase = song["measures"][start_measure:start_measure + 4]
            full_sequence = create_data_dict()
            for i, measure in enumerate(four_bar_phrase):
                for j, group in enumerate(measure["groups"]):
                    chord_root = group[const.CHORD_KEY]['root']
                    chord_type = group[const.CHORD_KEY]['type']
                    # print('chord_root: {}, chord_type: {}'.format(chord_root, chord_type))
                    full_sequence[const.CHORD_ROOT_KEY].extend([chord_root for _ in range(len(group['ticks']))])
                    full_sequence[const.CHORD_TYPE_KEY].extend([chord_type for _ in range(len(group['ticks']))])

                    formatted_ticks = []
                    for tick in group['ticks']:
                        # formatted = tick[range_low - 1:range_high + 1]
                        if tick == 0 or tick < A0 or tick > C8:
                            formatted = 0
                        else:
                            formatted = tick - A0 + 1 # lower bound maps to 1, not 0 (rest)
                        formatted_ticks.append(formatted)

                    full_sequence[const.TICK_KEY].extend(formatted_ticks)

            assert len(full_sequence[const.TICK_KEY]) == SEQ_LEN * 4
            write_to_rl_hdf5(h5_rl_fp, full_sequence[const.TICK_KEY], full_sequence[const.CHORD_ROOT_KEY],
                    full_sequence[const.CHORD_TYPE_KEY])

            start_measure += 4

        file_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction and HDF5 Dataset Creation')
    parser.add_argument('-d', '--dataset', choices=("charlie_parker", "bebop", "nottingham"), type=str, 
                        required=True, help="the dataset to use.")
    parser.add_argument('-ms', '--measures_per_seq', default=4, type=int, 
                        help="how many measures to include in a full sequence.")
    parser.add_argument('-hs', '--hop_size', default=4, type=int, 
                        help="how many measures to skip in between when extracting sequences.")
    parser.add_argument('-rl', '--range_low', default=A0, type=int, 
                        help="all midi numbers below this number are removed. Default is A0 (21)")
    parser.add_argument('-rh', '--range_high', default=C8, type=int, 
                        help="all midi numbers above this number are removed. Default is C8 (108)")
    args = parser.parse_args()

    main(args)
