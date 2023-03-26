import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import random

import numpy as np
from torch.utils.data import Dataset

import h5py

from models import UNet3D, VNet
from CustomImageDataset import h5_dataset

DATE_FORMAT = '%Y-%m-%d_%H-%m-%S'

generator = torch.Generator()
generator.manual_seed(0)


h5_name = 'vnet_CrossEntropy_AUGMENTATION-2-4000-pred-epochs-300.h5'

def predict_set(model, model_name, out_dir, label=None, set_size=799,
                predict_only="True", h5_file_path=None, train_part=0.8):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    h5_path = os.path.join(out_dir, h5_name)

    def predict(data_loader, model, label, out_dir):
        with h5py.File(h5_path, 'w') as out_file:
            with torch.no_grad():
                for i, (input, y_true) in enumerate(data_loader):
                    input = input.float().to(torch.device('cuda:0'))
                    pred = model(input)

                    input = np.squeeze(input.cpu().numpy())
                    y_true = np.squeeze(y_true.cpu().numpy())

                    background = pred[:,0].cpu().numpy()
                    labels = pred[:,1].cpu().numpy()
#                   pred = np.squeeze(pred.cpu().numpy())

                    data = np.stack((input, background, labels, y_true), axis=0)
#                   data = np.stack((input, pred, y_true), axis=0)
                    out_file.create_dataset(str(i), data=data,
                               shape=data.shape, dtype=np.float32)

    def create_loaders(h5_file_path, train_part):

        dataset = h5_dataset(h5_file_path)
        train_part = round(len(dataset) * train_part)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                    [train_part, len(dataset) - train_part],
                                                                    generator=generator)

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        return train_dataloader, test_dataloader

    if model == 'Unet3D':
        model = UNet3D(1, 2, final_sigmoid=False, f_maps=64, layer_order='cr',
                 num_levels=4, is_segmentation=False, conv_padding=1).to(torch.device('cuda:0'))
    else:
        model = VNet(1, 2).to(torch.device('cuda:0'))
#       model = VNet(1, 1).to(torch.device('cuda:0'))

    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(torch.device('cuda:0'))

    if predict_only == "True":
        train_dataloader, test_dataloader = create_loaders(h5_file_path, train_part)
#       predict(train_dataloader, model, 'train', out_dir)
        predict(test_dataloader, model, 'test', out_dir)
    else:
        predict(data_loader, model, label, out_dir)

