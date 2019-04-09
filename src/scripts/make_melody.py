import argparse
import copy
import json
import numpy as np
import os
import os.path as op
import pickle
import random
import sys
import torch
from torch.autograd import Variable
from pathlib import Path

PITCH_NUM_OFFSET = 21 # A0
SEQ_LEN = 64 # Hack for now 02/08/19

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from models.no_cond.model_classes import PitchLSTM as NoCondPitch, DurationLSTM as NoCondDur
from models.inter_cond.model_classes import PitchLSTM as InterCondPitch, DurationLSTM as InterCondDur
from models.barpos_cond.model_classes import PitchLSTM as BarPosCondPitch, DurationLSTM as BarPosCondDur
from models.chord_cond.model_classes import PitchLSTM as ChordCondPitch, DurationLSTM as ChordCondDur
from models.nxt_chord_cond.model_classes import PitchLSTM as NxtCondPitch, DurationLSTM as NxtCondDur
from models.chord_nxtchord_cond.model_classes import PitchLSTM as ChordNxtCondPitch, DurationLSTM as ChordNxtCondDur
from models.chord_inter_cond.model_classes import (
        PitchLSTM as ChordInterCondPitch, 
        DurationLSTM as ChordInterCondDur)
from models.chord_nxtchord_inter_cond.model_classes import (
        PitchLSTM as ChordNxtInterCondPitch, 
        DurationLSTM as ChordNxtInterCondDur)
from models.chord_barpos_cond.model_classes import (
        PitchLSTM as ChordBarPosCondPitch, 
        DurationLSTM as ChordBarPosCondDur)
from models.chord_nxtchord_barpos_cond.model_classes import (
        PitchLSTM as ChordNxtBarPosCondPitch, 
        DurationLSTM as ChordNxtBarPosCondDur)
from models.inter_barpos_cond.model_classes import (
        PitchLSTM as InterBarPosCondPitch, 
        DurationLSTM as InterBarPosCondDur)
from models.chord_inter_barpos_cond.model_classes import (
        PitchLSTM as ChordInterBarPosCondPitch, 
        DurationLSTM as ChordInterBarPosCondDur)
from models.chord_nxtchord_inter_barpos_cond.model_classes import (
        PitchLSTM as ChordNxtInterBarPosCondPitch, 
        DurationLSTM as ChordNxtInterBarPosCondDur)

import utils.constants as const
from utils.reverse_pianoroll import piano_roll_to_pretty_midi

ABRV_TO_MODEL = {'nc': {'name': 'no_cond',
                        'pitch_model': NoCondPitch,
                        'duration_model': NoCondDur}, 
                 'ic': {'name': 'inter_cond',
                        'pitch_model': InterCondPitch,
                        'duration_model': InterCondDur},
                 'bc': {'name': 'barpos_cond', 
                        'pitch_model': BarPosCondPitch,
                        'duration_model': BarPosCondDur},
                 'cc': {'name': 'chord_cond', 
                        'pitch_model': ChordCondPitch,
                        'duration_model': ChordCondDur},
                 'nxc': {'name': 'nxt_chord_cond', 
                        'pitch_model': NxtCondPitch,
                        'duration_model': NxtCondDur},
                 'cnc': {'name': 'chord_nxtchord_cond', 
                        'pitch_model': ChordNxtCondPitch,
                        'duration_model': ChordNxtCondDur},
                 'cic': {'name': 'chord_inter_cond',
                        'pitch_model': ChordInterCondPitch,
                        'duration_model': ChordInterCondDur},
                 'cnic': {'name': 'chord_nxtchord_inter_cond',
                        'pitch_model': ChordNxtInterCondPitch,
                        'duration_model': ChordNxtInterCondDur},
                 'cbc': {'name': 'chord_barpos_cond',
                        'pitch_model': ChordBarPosCondPitch,
                        'duration_model': ChordBarPosCondDur},
                 'cnbc': {'name': 'chord_nxtchord_barpos_cond',
                        'pitch_model': ChordNxtBarPosCondPitch,
                        'duration_model': ChordNxtBarPosCondDur},
                 'ibc': {'name': 'inter_barpos_cond',
                        'pitch_model': InterBarPosCondPitch,
                        'duration_model': InterBarPosCondDur},
                 'cibc': {'name': 'chord_inter_barpos_cond',
                        'pitch_model': ChordInterBarPosCondPitch,
                        'duration_model': ChordInterBarPosCondDur},
                 'cnibc': {'name': 'chord_nxtchord_inter_barpos_cond',
                        'pitch_model': ChordNxtInterBarPosCondPitch,
                        'duration_model': ChordNxtInterBarPosCondDur}}

CHORD_OFFSET = 48 # chords will be in octave 3
# BUFF_LEN = 32

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default="generated", type=str,
                    help="what to name the output")
parser.add_argument('-m', '--model', type=str, choices=list(ABRV_TO_MODEL.keys()),
                    help="which model to use for generation.\n \
                          \t\tnc - no_cond, ic - inter_cond, bc - barpos_cond, cc - chord_cond, \n \
                          \t\tnxc - next_chord_cond, cnc - chord_nextchord_cond, cic - chord_inter__cond, \n \
                          \t\tcnic - chord_nextchord_inter_cond, cbc - chord_barpos_cond, \n \
                          \t\tcnbc - chord_nextchord_barpos_cond, ibc - inter_barpos_cond \n \
                          \t\tcibc - chord_inter_barpos_cond, cnibc - chord_nextchord_inter_barpos_cond.")
parser.add_argument('-pn', '--pitch_run_name', type=str,
                    help="select which pitch run to use")
parser.add_argument('-dn', '--dur_run_name', type=str,
                    help="select which dur run to use")
parser.add_argument('-ss', '--gen_song', type=str, default=None,
                    help="which song to generate over.")
# parser.add_argument('-sm', '--seed_measures', type=int, default=1,
#                     help="number of measures to use as seeds to the network")
parser.add_argument('-sl', '--seed_len', type=int, default=3,
                    help="number of events to use for the seed")
parser.add_argument('-nr', '--num_repeats', type=int, default=1,
                    help="generate over the chords (nr) times.")
args = parser.parse_args()


def convert_melody_to_piano_roll_mat(pitches, dur_nums):
    dur_ticks = [const.DUR_TICKS_MAP[const.REV_DURATIONS_MAP[dur]] for dur in dur_nums]
    onsets = np.array([np.sum(dur_ticks[:i]) for i in range(len(dur_ticks))])
    total_ticks = sum(dur_ticks)
    output_mat = np.zeros([128, int(total_ticks)])
    for i in range(len(pitches) - 1):
        if pitches[i - 1] == const.NOTES_MAP['rest']:
            continue
        else:
            # include the -1 for now because stuff is out of key
            output_mat[int(pitches[i - 1]) + PITCH_NUM_OFFSET, int(onsets[i]):int(onsets[i+1])] = 1.0
    output_mat[int(pitches[-1] - 1) + PITCH_NUM_OFFSET, int(onsets[-1]):] = 1.0
    return output_mat


def convert_chords_to_piano_roll_mat(song_structure):
    output_mat = np.zeros([128, len(song_structure)*const.DUR_TICKS_MAP['whole']])
    for i, measure in enumerate(song_structure):
        ticks = i*const.DUR_TICKS_MAP['whole']
        for j, group in enumerate(measure):
            _, _, _, _, chord_vec, begin, end = group
            chord_block = np.array(chord_vec).reshape((len(chord_vec), 1)).repeat(end - begin, axis=1)
            output_mat[CHORD_OFFSET:CHORD_OFFSET + len(chord_vec), ticks + begin:ticks + end] = chord_block
    return output_mat


def roll(x, n):  
    return torch.cat((x[-n:], x[:-n]))

def step(model_abrv, pitch_net, dur_net, pitch_inpt, dur_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt, 
        nxt_chord_root_inpt, nxt_chord_pc_inpt):
        if args.model == "nc":
            pitch_out = pitch_net(pitch_inpt)
            dur_out = dur_net(dur_inpt)
        elif args.model == "ic":
            pitch_out = pitch_net((pitch_inpt, dur_inpt))
            dur_out = dur_net((dur_inpt, pitch_inpt))
        elif args.model == "bc":
            pitch_out = pitch_net((pitch_inpt, barpos_inpt))
            dur_out = dur_net((dur_inpt, barpos_inpt))
        elif args.model == "cc":
            pitch_out = pitch_net((pitch_inpt, chord_root_inpt, chord_pc_inpt))
            dur_out = dur_net((dur_inpt, chord_root_inpt, chord_pc_inpt))
        elif args.model == "nxc":
            pitch_out = pitch_net((pitch_inpt, nxt_chord_root_inpt, nxt_chord_pc_inpt))
            dur_out = dur_net((dur_inpt, nxt_chord_root_inpt, nxt_chord_pc_inpt))
        elif args.model == "cnc":
            pitch_out = pitch_net((pitch_inpt, chord_root_inpt, chord_pc_inpt, nxt_chord_root_inpt, nxt_chord_pc_inpt))
            dur_out = dur_net((dur_inpt, chord_root_inpt, chord_pc_inpt, nxt_chord_root_inpt, nxt_chord_pc_inpt))
        elif args.model == "cic":
            pitch_out = pitch_net((pitch_inpt, dur_inpt, chord_root_inpt, chord_pc_inpt))
            dur_out = dur_net((dur_inpt, pitch_inpt, chord_root_inpt, chord_pc_inpt))
        elif args.model == "cnic":
            pitch_out = pitch_net((pitch_inpt, dur_inpt, chord_root_inpt, chord_pc_inpt,
                                    nxt_chord_root_inpt, nxt_chord_pc_inpt))
            dur_out = dur_net((dur_inpt, pitch_inpt, chord_root_inpt, chord_pc_inpt,
                                nxt_chord_root_inpt, nxt_chord_pc_inpt))
        elif args.model == "cbc":
            pitch_out = pitch_net((pitch_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt))
            dur_out = dur_net((dur_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt))
        elif args.model == "cnbc":
            pitch_out = pitch_net((pitch_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt,
                                    nxt_chord_root_inpt, nxt_chord_pc_inpt))
            dur_out = dur_net((dur_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt,
                                nxt_chord_root_inpt, nxt_chord_pc_inpt))
        elif args.model == "ibc":
            pitch_out = pitch_net((pitch_inpt, dur_inpt, barpos_inpt))
            dur_out = dur_net((dur_inpt, pitch_inpt, barpos_inpt))
        elif args.model == "cibc":
            pitch_out = pitch_net((pitch_inpt, dur_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt))
            dur_out = dur_net((dur_inpt, pitch_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt))
        elif args.model == "cnibc":
            pitch_out = pitch_net((pitch_inpt, dur_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt,
                                    nxt_chord_root_inpt, nxt_chord_pc_inpt))
            dur_out = dur_net((dur_inpt, pitch_inpt, barpos_inpt, chord_root_inpt, chord_pc_inpt,
                                nxt_chord_root_inpt, nxt_chord_pc_inpt))

        return pitch_out, dur_out

# prepare models
root_dir = str(Path(op.abspath(__file__)).parents[2])
model_dir = op.join(root_dir, 'src', 'models', ABRV_TO_MODEL[args.model]['name'])

pitch_dir = op.join(model_dir, 'runs', 'pitch', args.pitch_run_name)
dur_dir = op.join(model_dir, 'runs', 'duration', args.dur_run_name)

pitch_model_inputs = json.load(open(op.join(pitch_dir, 'model_inputs.json'), 'r'))
pitch_model_inputs['batch_size'] = 1
# pitch_model_inputs['seq_len'] = 1
pitch_model_inputs['dropout'] = 0
pitch_model_inputs['no_cuda'] = True
pitch_model_state = torch.load(op.join(pitch_dir, 'model_state.pt'), map_location='cpu')

PitchModel = ABRV_TO_MODEL[args.model]['pitch_model'](**pitch_model_inputs)
PitchModel.load_state_dict(pitch_model_state)
PitchModel.eval()

dur_model_inputs = json.load(open(op.join(dur_dir, 'model_inputs.json'), 'r'))
dur_model_inputs['batch_size'] = 1
# dur_model_inputs['seq_len'] = 1
dur_model_inputs['dropout'] = 0
dur_model_inputs['no_cuda'] = True
dur_model_state = torch.load(op.join(dur_dir, 'model_state.pt'), map_location='cpu')

DurModel = ABRV_TO_MODEL[args.model]['duration_model'](**dur_model_inputs)
DurModel.load_state_dict(dur_model_state)
DurModel.eval()

PitchModel.init_hidden_and_cell(1)
DurModel.init_hidden_and_cell(1)

### Set up the seed and song structure
data_dir = op.join(root_dir, 'data', 'processed', 'songs')
if args.gen_song is None:
    songs = os.listdir(data_dir)
    gen_song = pickle.load(open(op.join(data_dir, random.choice(songs)), 'rb'))
else:
    gen_song = pickle.load(open(op.join(data_dir, args.gen_song), 'rb'))

## Get Song Structure
# these two will allow the generation of a new melody by keeping track of within measure
# chord and bar positions.
song_structure = []
for i, measure in enumerate(gen_song["measures"]):
    measure_groups = []
    for j, group in enumerate(measure["groups"]):
        chord = group["harmony"]["root"] + group["harmony"]["pitch_classes"]
        chord_root = np.argmax(group["harmony"]["root"])
        chord_pc = group["harmony"]["pitch_classes"]
        nxt_chord_root = np.argmax(group["next_harmony"]["root"])
        nxt_chord_pc = group["next_harmony"]["pitch_classes"]
        begin = group[const.BARPOS_KEY][0]
        end = group[const.BARPOS_KEY][-1] + const.DUR_TICKS_MAP[const.REV_DURATIONS_MAP[group[const.DUR_KEY][-1]]]
       
        measure_groups.append((chord_root, chord_pc, nxt_chord_root, nxt_chord_pc, chord, begin, end))
    song_structure.append(measure_groups)

## Seed The Model
seed_data = {const.PITCH_KEY: [], 
             const.DUR_KEY: [], 
             const.BARPOS_KEY: [],
             const.CHORD_ROOT_KEY: [], 
             const.CHORD_PC_KEY: [], 
             const.NXT_CHORD_ROOT_KEY: [], 
             const.NXT_CHORD_PC_KEY: []}

for i, measure in enumerate(gen_song["measures"]):
    for j, group in enumerate(measure["groups"]):
        assert len(group[const.PITCH_KEY]) == len(group[const.DUR_KEY]) == len(group[const.BARPOS_KEY])
        # so the seed is sort of random each time
        pitch_group = copy.deepcopy(group[const.PITCH_KEY])
        random.shuffle(pitch_group)

        seed_data[const.PITCH_KEY].extend(pitch_group) # for variability
        seed_data[const.DUR_KEY].extend(group[const.DUR_KEY])
        seed_data[const.BARPOS_KEY].extend(group[const.BARPOS_KEY])
        seed_data[const.CHORD_ROOT_KEY].extend([np.argmax(group["harmony"]["root"])] * len(group[const.PITCH_KEY]))
        seed_data[const.CHORD_PC_KEY].extend([group["harmony"]["pitch_classes"]] * len(group[const.PITCH_KEY]))
        seed_data[const.NXT_CHORD_ROOT_KEY].extend([np.argmax(group["next_harmony"]["root"])] * len(group[const.PITCH_KEY]))
        seed_data[const.NXT_CHORD_PC_KEY].extend([group["next_harmony"]["pitch_classes"]] * len(group[const.PITCH_KEY]))

    if len(seed_data[const.PITCH_KEY]) > args.seed_len:
        break

pitch_inpt = torch.LongTensor(np.zeros([SEQ_LEN]))
dur_inpt = torch.LongTensor(np.zeros([SEQ_LEN]))
barpos_inpt = torch.LongTensor(np.zeros([SEQ_LEN]))
chord_root_inpt = torch.LongTensor(np.zeros([SEQ_LEN]))
chord_pc_inpt = torch.FloatTensor(np.zeros([SEQ_LEN, const.CHORD_PC_DIM])) 
nxt_chord_root_inpt = torch.LongTensor(np.zeros([SEQ_LEN]))
nxt_chord_pc_inpt = torch.FloatTensor(np.zeros([SEQ_LEN, const.CHORD_PC_DIM])) 

seed_data = {k: np.array(v[:args.seed_len]) for k, v in seed_data.items()}
pitch_inpt[-args.seed_len:] = torch.LongTensor(seed_data[const.PITCH_KEY])
dur_inpt[-args.seed_len:] = torch.LongTensor(seed_data[const.DUR_KEY])
barpos_inpt[-args.seed_len:] = torch.LongTensor(seed_data[const.BARPOS_KEY])
chord_root_inpt[-args.seed_len:] = torch.LongTensor(seed_data[const.CHORD_ROOT_KEY])
chord_pc_inpt[-args.seed_len:] = torch.LongTensor(seed_data[const.CHORD_PC_KEY])
nxt_chord_root_inpt[-args.seed_len:] = torch.LongTensor(seed_data[const.NXT_CHORD_ROOT_KEY])
nxt_chord_pc_inpt[-args.seed_len:] = torch.LongTensor(seed_data[const.NXT_CHORD_PC_KEY])

# for i in range(len(seed_data[const.PITCH_KEY])):
#     pitch_inpt = torch.LongTensor([seed_data[const.PITCH_KEY][i]]).view(1, -1)
#     dur_inpt = torch.LongTensor([seed_data[const.DUR_KEY][i]]).view(1, -1)
#     barpos_inpt = torch.LongTensor([seed_data[const.BARPOS_KEY][i]]).view(1, -1)
#     chord_root_inpt = torch.LongTensor([seed_data[const.CHORD_ROOT_KEY][i]]).view(1, -1)
#     chord_pc_inpt = torch.FloatTensor(seed_data[const.CHORD_PC_KEY][i]).view(1, -1, const.CHORD_PC_DIM)
#     nxt_chord_root_inpt = torch.LongTensor([seed_data[const.NXT_CHORD_ROOT_KEY][i]]).view(1, -1)
#     nxt_chord_pc_inpt = torch.FloatTensor(seed_data[const.NXT_CHORD_PC_KEY][i]).view(1, -1, const.CHORD_PC_DIM)
    
#     step(args.model, PitchModel, DurModel, pitch_inpt, dur_inpt, barpos_inpt, 
#         chord_root_inpt, chord_pc_inpt, nxt_chord_root_inpt, nxt_chord_pc_inpt)

pitch_seq = []
dur_seq = []
barpos_seq = []
harmony_seq = []
for _ in range(args.num_repeats):
    for i, measure in enumerate(song_structure):
        curr_barpos = 0
        for j, group in enumerate(measure):
            chord_root, chord_pc, nxt_chord_root, nxt_chord_pc, chord, begin, end = group
            while curr_barpos < end:
                pitch_out, dur_out = step(args.model, PitchModel, DurModel, pitch_inpt.view(1, -1), dur_inpt.view(1, -1), 
                                          barpos_inpt.view(1, -1), chord_root_inpt.view(1, -1), 
                                          chord_pc_inpt.view(1, -1, const.CHORD_PC_DIM), nxt_chord_root_inpt.view(1, -1), 
                                          nxt_chord_pc_inpt.view(1, -1, const.CHORD_PC_DIM))

                pitch_seq.append(int(torch.exp(pitch_out.data[:, -1, :]).multinomial(1)))
                dur_seq.append(int(torch.exp(dur_out.data[:, -1, :]).multinomial(1)))
                barpos_seq.append(curr_barpos)
                harmony_seq.append(chord)

                pitch_inpt = roll(pitch_inpt, -1)
                pitch_inpt[-1] = torch.LongTensor([pitch_seq[-1]])

                dur_inpt = roll(dur_inpt, -1)
                dur_inpt[-1] = torch.LongTensor([dur_seq[-1]])

                barpos_inpt = roll(barpos_inpt, -1)
                barpos_inpt[-1] = torch.LongTensor([barpos_seq[-1]])

                chord_root_inpt = roll(chord_root_inpt, -1)
                chord_root_inpt[-1] = torch.LongTensor([chord_root])

                chord_pc_inpt = roll(chord_pc_inpt, -1)
                chord_pc_inpt[-1] = torch.FloatTensor(chord_pc)

                nxt_chord_root_inpt = roll(nxt_chord_root_inpt, -1)
                nxt_chord_root_inpt[-1] = torch.LongTensor([nxt_chord_root])

                nxt_chord_pc_inpt = roll(nxt_chord_pc_inpt, -1)
                nxt_chord_pc_inpt[-1] = torch.FloatTensor(nxt_chord_pc)
                # pitch_inpt = torch.LongTensor([pitch_seq[-1]]).view(1, -1)
                # dur_inpt = torch.LongTensor([dur_seq[-1]]).view(1, -1)
                # barpos_inpt = torch.LongTensor([barpos_seq[-1]]).view(1, -1)
                # chord_root_inpt = torch.LongTensor(chord_root).view(1, -1)
                # chord_pc_inpt = torch.FloatTensor(chord_pc).view(1, -1, const.CHORD_PC_DIM)
                # nxt_chord_root_inpt = torch.LongTensor(nxt_chord_root).view(1, -1)
                # nxt_chord_pc_inpt = torch.FloatTensor(nxt_chord_pc).view(1, -1, const.CHORD_PC_DIM)

                PitchModel.init_hidden_and_cell(1)
                DurModel.init_hidden_and_cell(1)
                curr_barpos += const.DUR_TICKS_MAP[const.REV_DURATIONS_MAP[dur_seq[-1]]]

outdir = op.join(model_dir, 'midi', args.title)
if not op.exists(outdir):
    os.makedirs(outdir)

tokens = {'pitch_numbers': [], 'duration_tags': []}
measure_pns = []
measure_dts = []
prev_barpos = -1
# print(barpos_seq)
for i, barpos in enumerate(barpos_seq):
    if barpos <= prev_barpos:
        tokens['pitch_numbers'].append(measure_pns)
        tokens['duration_tags'].append(measure_dts)
        measure_pns = []
        measure_dts = []
    measure_pns.append(pitch_seq[i])
    measure_dts.append(dur_seq[i])
    prev_barpos = barpos
tokens['pitch_numbers'].append(measure_pns)
tokens['duration_tags'].append(measure_dts)

tokens_path = op.join(outdir, "%s_tokens.json" % args.title);
print('Writing tokens file %s ...' % tokens_path)
json.dump(tokens, open(tokens_path, 'w'))
        
melody_pr_mat = convert_melody_to_piano_roll_mat(pitch_seq, dur_seq)
chords_pr_mat = convert_chords_to_piano_roll_mat(song_structure)
melody_pm = piano_roll_to_pretty_midi(melody_pr_mat, fs=30)
chords_pm = piano_roll_to_pretty_midi(chords_pr_mat, fs=30)
melody_path = op.join(outdir, '%s_melody.mid' % args.title)
chords_path = op.join(outdir, '%s_chords.mid' % args.title)
print('Writing melody midi file %s ...' % melody_path)
melody_pm.write(melody_path)
print('Writing chords midi file %s ...' % chords_path)
chords_pm.write(chords_path)
