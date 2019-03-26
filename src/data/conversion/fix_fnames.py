import os
import os.path as op
from pathlib import Path

root_dir = str(Path(op.abspath(__file__)).parents[3])
raw_data_dir = op.join(root_dir, 'data', 'raw')
raw_data_paths = [op.join(raw_data_dir, name) for name in os.listdir(raw_data_dir)
                  if name != '.DS_Store']

for path in raw_data_paths:
    print(path)
    fnames = os.listdir(path)
    for fname in fnames:
        print('fixing %s ...' % op.join(op.basename(path), fname))
        parts = fname.split('.')
        name = ''.join(parts[:-1])
        ext = parts[-1]

        try:        
            artist, song = name.split('-')
        except ValueError:
            print('\texception! not fixed')
            continue
        fixed_name = "-".join([artist.lower().strip().replace(' ', '_'),
                               song.lower().strip().replace(' ', '_')])
        fixed_fname = ".".join([fixed_name, ext])
        os.rename(op.join(path, fname), op.join(path, fixed_fname))
