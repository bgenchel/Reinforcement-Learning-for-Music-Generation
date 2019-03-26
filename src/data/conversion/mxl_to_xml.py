import os
import os.path as op
import subprocess
from pathlib import Path

def main():
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    mxl_dir = op.join(root_dir, 'data', 'raw', 'mxl')

    fnames = os.listdir(mxl_dir)
    fdicts = [{'basename': fname.split('.')[0], 
               'path': op.join(mxl_dir, fname)} for fname in fnames]

    for fdict in fdicts:
        outpath = op.join(root_dir, 'data', 'raw', 'xml', fdict['basename'] + '.xml')
        mscore_path = '/Applications/MuseScore 2.app/Contents/MacOS/mscore'
        subprocess.call([mscore_path, fdict['path'], '-o', outpath])
    return

if __name__ == '__main__':
    main()
