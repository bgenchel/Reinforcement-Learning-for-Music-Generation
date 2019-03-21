"""
Constant Values
"""
# Strings
TICK_KEY = "ticks"
SEQS_KEY = "sequences"
CR_SEQS_KEY = "chord_root_sequences"
CT_SEQS_KEY = "chord_type_sequences"
TARGETS_KEY = "targets"

### From Explicitly Conditioning Melody Paper
PITCH_KEY = "pitch_numbers"
DUR_KEY = "duration_tags"
CHORD_KEY = "harmony"
NXT_CHORD_KEY = "next_harmony"
CHORD_ROOT_KEY = "harmony_root"
CHORD_TYPE_KEY = "harmony_type"
CHORD_PC_KEY = "harmony_pitch_classes"
NXT_CHORD_ROOT_KEY = "next_harmony_root"
NXT_CHORD_PC_KEY = "next_harmony_pitch_classes"
BARPOS_KEY = "bar_positions"

# Model Params
PITCH_DIM = 89
DUR_DIM = 19
CHORD_DIM = 24
CHORD_ROOT_DIM = 12
CHORD_PC_DIM = 12
BARPOS_DIM = 96
PITCH_EMBED_DIM = 8
DUR_EMBED_DIM = 4
CHORD_ROOT_EMBED_DIM = 2
CHORD_PC_EMBED_DIM = 4
CHORD_EMBED_DIM = 8
BARPOS_EMBED_DIM = 8
NUM_RNN_LAYERS = 2
TICKS_PER_MEASURE = 96

# Dictionaries
NOTES_MAP = {'rest': 88, 'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 
             'Eb': 3, 'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6, 
             'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}

INVERSE_NOTES_MAP = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
    12: 'rest'
}

DURATIONS_MAP = {
    '32nd-triplet': 0, 
    '32nd': 1, 
    '16th-triplet': 2,
    '32nd-dot': 3,
    '16th': 4,
    '8th-triplet': 5,
    '16th-dot': 6,
    '8th': 7,
    'quarter-triplet': 8,
    '8th-dot': 9,
    'quarter': 10,
    'half-triplet': 11,
    'quarter-dot': 12,
    'half': 13,
    'whole-triplet': 14,
    'half-dot': 15,
    'whole': 16,
    'double-triplet': 17,
    'whole-dot': 18,
    'double': 19,
    'double-dot': 20,
    'none': 21
}

REV_DURATIONS_MAP = {v: k for k, v in DURATIONS_MAP.items()}

# 96 ticks per bar
DUR_TICKS_MAP = {
    '32nd-triplet': 2, 
    '32nd': 3, 
    '16th-triplet': 4,
    '32nd-dot': 5,
    '16th': 6,
    '8th-triplet': 8,
    '16th-dot': 9,
    '8th': 12,
    'quarter-triplet': 16,
    '8th-dot': 18,
    'quarter': 24,
    'half-triplet': 32,
    'quarter-dot': 36,
    'half': 48,
    'whole-triplet': 64,
    'half-dot': 72,
    'whole': 96,
    'double-triplet': 128,
    'whole-dot': 144,
    'double': 192,
    'double-dot': 288,
    'none': 0
}

KEYS_DICT = {"major": {'0': 'C', '1': 'G', '2': 'D', '3': 'A', '4': 'E', '5': 'B', '6': 'F#', 
                  '-1': 'F', '-2': 'Bb', '-3': 'Eb', '-4': 'Ab', '-5': 'Db', '-6': 'Gb'},
            "minor": {'0': 'A', '1': 'E', '2': 'B', '3': 'F#', '4': 'C#', '5': 'G#', '6': 'D#', 
                  '-1': 'D', '-2': 'G', '-3': 'C', '-4': 'F', '-5': 'Bb', '-6': 'Eb'}}
