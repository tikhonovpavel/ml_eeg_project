import os
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--h5_filename_to_unarchive", help="specify name of h5 archived dataset")
args = parser.parse_args()

archive = h5py.File(args.h5_filename_to_unarchive, 'r')
keys = list(archive.keys())

unarchived_file_name = args.h5_filename_to_unarchive.replace('.h5', '_unarch.h5')


with h5py.File(unarchived_file_name, 'w') as out_file:
    for data_name in archive:
        dataset = archive.get(data_name)
        out_file.create_dataset(data_name, data=dataset,
                                shape=dataset.shape, dtype=np.float32)
