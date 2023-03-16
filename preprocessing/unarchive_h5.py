import h5py
import numpy as np
import argparse

# create argument parser to accept filename
parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="path to archived h5 file")
args = parser.parse_args()

# use path to archived h5
h5_file = h5py.File(args.filename, 'r')
dataset = h5_file.get('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000')
storage = np.array(dataset)

# create unarchived h5 file
unarchived_file_name = args.filename.replace('.h5', '_unarch.h5')
hf_unarchived = h5py.File(unarchived_file_name, 'w')
hf_unarchived.create_dataset('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000', data=storage)
hf_unarchived.close()
