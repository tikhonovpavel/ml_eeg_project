import os
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_size", type=int, default=4000, help="path to voxelized eeg files")
parser.add_argument("--electrodes_dir", help="path to voxelized eeg files")
parser.add_argument("--dipoles_dir", help="path to voxelized dipoles files")
parser.add_argument("--h5_filename_to_save", help="specify name of h5 archived dataset")
args = parser.parse_args()


with h5py.File(args.h5_filename_to_save, 'w') as out_file:
    for i, (img, label) in enumerate(list(zip(os.listdir(args.electrodes_dir), os.listdir(args.dipoles_dir)))):
                
        electrodes_npz = np.load(os.path.join(electrodes_dir, img))
        electrodes = np.expand_dims(electrodes_npz['arr_0'], axis=0)

        dipoles_npz =  np.load(os.path.join(dipoles_dir, label))
        dipoles = np.expand_dims(dipoles_npz['arr_0'], axis=0)

        data = np.concatenate((electrodes, dipoles), axis=0)
        data_name = img.split(".")[0]

        out_file.create_dataset(data_name, data=data,
                                shape=data.shape, dtype=np.float32, compression="gzip")
        
        if i+1 == args.dataset_size: break
