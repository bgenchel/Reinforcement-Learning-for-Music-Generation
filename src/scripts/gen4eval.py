import argparse
import glob
import os
import os.path as op
import subprocess
from collections import OrderedDict
from pathlib import Path


def generate_songs(model_abrv, model_name, seed_length, dataset, songs):
    print("#"*30 + "\n{}\n".format(model_name) + "#"*30)
    for i, song in enumerate(songs):
        outname = "_".join(["4eval", dataset, song.split('.')[0]])
        try:
            subprocess.call(['python', 'make_melody.py', '-m', model_abrv,  
                             '-pn', args.pitch_run_name, '-dn', args.dur_run_name,  
                             '-ss', song, '-sl', str(args.seed_length), '-t', outname])
        except RuntimeError:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="all", choices=(list(ABRV_TO_MODEL.keys()) + ['all']),
                        help="which model to use for generation.\n \
                              \t\tall, nc - no_cond, ic - inter_cond, bc - barpos_cond, cc - chord_cond, \n \
                              \t\tnxc - next_chord_cond, cnc - chord_nextchord_cond, cic - chord_inter__cond, \n \
                              \t\tcnic - chord_nextchord_inter_cond, cbc - chord_barpos_cond, \n \
                              \t\tcnbc - chord_nextchord_barpos_cond, ibc - inter_barpos_cond \n \
                              \t\tcibc - chord_inter_barpos_cond, cnibc - chord_nextchord_inter_barpos_cond.")
    parser.add_argument('-pn', '--pitch_run_name', type=str, default="MUME_Bebop-Feb12",
                        help="select which pitch run to use")
    parser.add_argument('-dn', '--dur_run_name', type=str, default="MUME_Bebop-Feb12",
                        help="select which dur run to use")
    parser.add_argument('-nr', '--num_repeats', type=int, default=1,
                        help="if you want to generate more than 1 run through of a song")
    parser.add_argument('-sl', '--seed_length', type=int, default=10,
                        help="number of measures to use as seeds to the network")
    parser.add_argument('-ds', '--dataset', type=str, choices=('Folk', 'Bebop'), required=True,
                        help="Which dataset to evaluate against.")
    args = parser.parse_args()

    root_dir = str(Path(op.abspath(__file__)).parents[2])
    model_dir = op.join(root_dir, "src", "models")
    data_song_dir = op.join(root_dir, "data", "processed", "songs")

    songs = [op.basename(s) for s in glob.glob(op.join(data_song_dir, '*_0.pkl'))]
    if args.model != 'all':
        generate_songs(args.model, ABRV_TO_MODEL[args.model], args.seed_length, args.dataset, songs)
    else:
        for abrv, name in ABRV_TO_MODEL.items():
            generate_songs(abrv, name, args.seed_length, args.dataset, songs)
