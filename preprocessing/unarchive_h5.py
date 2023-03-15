import h5py
import numpy as np

# use path to archived h5
h5_file = h5py.File('C:/Users/spaik/Documents/lab/ML_project/dense-162dip_parcell-64_GRID-64_paired_scale-False-2000.h5', 'r')
dataset = h5_file.get('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000')
storage = np.array(dataset)

# create unarchived h5 file
hf_unarchived = h5py.File('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000_unarch.h5', 'w')
hf_unarchived.create_dataset('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000', data=storage)
hf_unarchived.close()
