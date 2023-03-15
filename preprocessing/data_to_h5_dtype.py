import random
import os
import numpy as np
import h5py

electrodes_dir = 'C:/Users/spaik/Documents/lab/ML_project/data/1.3.23_dense_dip/162-dip__ico2/Parcellation_64-lbl/train/GRID-64/paired_scale-False/AUGMENTATION-None/electrodes'
dipoles_dir = 'C:/Users/spaik/Documents/lab/ML_project/data/1.3.23_dense_dip/162-dip__ico2/Parcellation_64-lbl/train/GRID-64/paired_scale-False/AUGMENTATION-None/dipoles'

data_size = 2000
dataset_input_storage = []
dataset_label_storage = []

random.seed(10)

counter = 0
while counter < data_size:
    img, label = random.choice(list(zip(os.listdir(electrodes_dir), os.listdir(dipoles_dir))))
    with np.load(os.path.join(electrodes_dir, img)) as img:
        image = np.expand_dims(img['arr_0'], axis=0)
        dataset_input_storage.append(image)
    with np.load(os.path.join(dipoles_dir, label)) as label:
        image = np.expand_dims(label['arr_0'], axis=0)
        dataset_label_storage.append(image)
    counter +=1

dataset_input_storage = np.asarray(dataset_input_storage)
dataset_label_storage = np.asarray(dataset_label_storage)

storage = np.concatenate((dataset_input_storage, dataset_label_storage), axis=1)

print('input + label storage shape: ' + str(storage.shape))

# varying compression_opts from 1 to 9 you will increase the degree of archivation
hf = h5py.File('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000.h5', 'w')
hf.create_dataset('dense-162dip_parcell-64_GRID-64_paired_scale-False-2000', data=storage, compression="gzip", compression_opts=1)
hf.close()
