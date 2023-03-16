from torch.utils.data import Dataset
import h5py
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, dataset_input_storage, dataset_label_storage, transform=None, target_transform=None):
        self.dataset_input_storage = dataset_input_storage
        self.dataset_label_storage = dataset_label_storage
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset_input_storage)

    def __getitem__(self, idx):
        image = self.dataset_input_storage[idx]
        label = self.dataset_label_storage[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
    
class h5_dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.dataset = self.h5_file.get(list(self.h5_file.keys())[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        electrodes = self.dataset[idx, 0]
        dipoles = self.dataset[idx, 1]

        electrodes = np.expand_dims(electrodes, axis=0)
        dipoles = np.expand_dims(dipoles, axis=0)

        return electrodes, dipoles
