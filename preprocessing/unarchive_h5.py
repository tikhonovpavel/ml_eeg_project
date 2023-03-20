import h5py
import argparse
import numpy as np




parser = argparse.ArgumentParser()
parser.add_argument("--filename", help="path to archived h5 file")
parser.add_argument("--chunk_size", default=500)
args = parser.parse_args()

h5_file = h5py.File(args.filename, 'r')
dataset = h5_file.get('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000')

chunk_size = 500
nrows = dataset.shape[0]
nchunks = (nrows - 1) // chunk_size + 1

unarchived_file_name = args.filename.replace('.h5', '_unarch.h5')
hf_unarchived = h5py.File(unarchived_file_name, 'w')

unarchived_dataset = hf_unarchived.create_dataset('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000',
                                                  shape=dataset.shape, dtype=dataset.dtype)

for i in range(nchunks):
    start = i * chunk_size
    end = min(start + chunk_size, nrows)
    chunk = dataset[start:end]
    unarchived_dataset[start:end] = chunk

h5_file.close()
hf_unarchived.close()
