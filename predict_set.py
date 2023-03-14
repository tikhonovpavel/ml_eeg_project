import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import random

import numpy as np
from torch.utils.data import Dataset


from models import UNet3D, VNet
from CustomImageDataset import CustomImageDataset

DATE_FORMAT = '%Y-%m-%d_%H-%m-%S'

def predict_set(model, model_name, out_dir, data_loader=None, label=None, set_size=10, data_limit=800,
                predict_only=False, img_dir=None, label_dir=None, train_part=None):
    '''
    for predict only write img_dir, label_dir, train_part into *dirs
    '''

    def predict(data_loader, model, label, out_dir):
        with torch.no_grad():
            for i, (input, y_true) in enumerate(data_loader):
                input = input.float().to(torch.device('cuda:0'))
                pred = model(input)
                np.savez(os.path.join(out_dir, f'{label}_{i}_pred'), input.cpu().numpy(), pred.cpu().numpy(), y_true)
                if i == set_size: break

    def create_loaders(img_dir, label_dir, train_part):
        dataset_input_storage = []
        dataset_label_storage = []
        
        random.seed(10)
        counter = 0
        while counter < data_limit:
            img, label = random.choice(list(zip(os.listdir(img_dir), os.listdir(label_dir))))
            with np.load(os.path.join(img_dir, img)) as img:
                image = np.expand_dims(img['arr_0'], axis=0)
                dataset_input_storage.append(image)
            with np.load(os.path.join(label_dir, label)) as label:
                image = np.expand_dims(label['arr_0'], axis=0)
                dataset_label_storage.append(image)
            counter +=1
        generator = torch.Generator() 
        generator.manual_seed(0)

        dataset = CustomImageDataset(dataset_input_storage, dataset_label_storage)
        train_part = round(len(dataset_input_storage) * train_part)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                    [train_part, len(dataset_input_storage) - train_part],
                                                                    generator=generator)
        
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        return  train_dataloader, test_dataloader

    if model == 'Unet3D':
        model = UNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='cr',
                 num_levels=4, is_segmentation=False, conv_padding=1).to(torch.device('cuda:0'))
    else:
        model = VNet(1, 1).to(torch.device('cuda:0'))

    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint)
    model = model.to(torch.device('cuda:0'))

    if predict_only:
        train_dataloader, test_dataloader = create_loaders(img_dir, label_dir, train_part)
        predict(train_dataloader, model, 'train', out_dir)
        predict(test_dataloader, model, 'test', out_dir)
    else:
        predict(data_loader, model, label, out_dir)


