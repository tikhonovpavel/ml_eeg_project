import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import random

import numpy as np
from torch.utils.data import Dataset

from models import UNet3D, VNet
from CustomImageDataset import H5Dataset

DATE_FORMAT = '%Y-%m-%d_%H-%m-%S'


def predict_set(model, model_name, out_dir, data_loader=None, label=None, set_size=10, data_limit=800,
                predict_only=False, h5_file_path=None, train_part=None):
    def predict(data_loader, model, label, out_dir):
        with torch.no_grad():
            for i, (input, y_true) in enumerate(data_loader):
                input = input.float().to(torch.device('cuda:0'))
                pred = model(input)
                np.savez(os.path.join(out_dir, f'{label}_{i}_pred'), input.cpu().numpy(), pred.cpu().numpy(), y_true)
                if i == set_size: break

    def create_loaders(h5_file_path, train_part):

        dataset = H5Dataset(h5_file_path)
        train_part = round(len(dataset) * train_part)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                    [train_part, len(dataset) - train_part],
                                                                    generator=generator)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader

    if model == 'Unet3D':
        model = UNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='cr',
                       num_levels=4, is_segmentation=False, conv_padding=1).to(torch.device('cuda:0'))
    else:
        model = VNet(1, 1).to(torch.device('cuda:0'))

    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint)
    model = model.to(torch.device('cuda:0'))

    if predict_only:
        train_dataloader, test_dataloader = create_loaders(h5_file_path, train_part)
        predict(train_dataloader, model, 'train', out_dir)
        predict(test_dataloader, model, 'test', out_dir)
    else:
        predict(data_loader, model, label, out_dir)
