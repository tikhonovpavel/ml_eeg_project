import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

import os
import random

import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm

from models import UNet3D, VNet

import psutil
import sys


OPTIMIZERS_LIST = ('Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'SGD', 'RAdam', 'Rprop',
                    'RMSprop', 'NAdam', 'LBFGS',)
LOSSES_LIST = ("CrossEntropyLoss", "BCELoss", "MSELoss", "L1Loss", "SmoothL1Loss", "KLDivLoss", "CosineEmbeddingLoss",
                "TripletMarginLoss", "HingeEmbeddingLoss", "MultiMarginLoss", )
DATE_FORMAT = '%Y-%m-%d_%H-%m-%S'

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


def start_training(label_dir, img_dir, out_dir, log_path, train_part, model, loss_fn, learning_rate, epochs, batch_size, optimizer, decay, data_limit):
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
    print(
        f'Dataset {len(dataset_input_storage)}\nTrain set : {train_part} images\nValidation set : {len(dataset_input_storage) - train_part} images')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(len(train_dataloader))

    # memory tests9
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print(f'memory GB:{memoryUse}')

    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 ** 3))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 ** 3))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 ** 3))

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")


    if model == 'Unet3D':
        model = UNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='cr',
                 num_levels=4, is_segmentation=False, conv_padding=1).to(device)
    else:
        model = VNet().to(device)
    
    print(model)

    if loss_fn in LOSSES_LIST:
        loss_fn = getattr(nn, loss_fn)()
    else:
        raise NotImplementedError()

    loss_fn = loss_fn.to(device)

    if optimizer in OPTIMIZERS_LIST:
        optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError()

    # logger inicialisation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # main train/val cycle
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(train_dataloader, model, optimizer, loss_fn, log_path, device)

        # We don't need gradients on to do reporting
        model.train(False)

        running_tloss = 0.
        for batch, (tinputs, tlabels) in tqdm(enumerate(test_dataloader)):
            tinputs, tlabels = tinputs.to(torch.device(device)), tlabels.to(torch.device(device))
            tinputs = tinputs.float()
            with torch.no_grad():
                toutputs = model(tinputs)

            tloss = loss_fn(toutputs, tlabels.float())
            running_tloss += tloss

        avg_tloss = running_tloss / (batch + 1)

        print('LOSS train {} test {}'.format(avg_loss, avg_tloss))

        with open(log_path, 'a') as log:
            log.write(f'{avg_tloss}\n')

        if epoch % 10 == 9:
            torch.save(model.state_dict(), os.path.join(out_dir, 'final_model_{}_{}.pt'.format(timestamp, epoch)))

    torch.save(model.state_dict(), os.path.join(out_dir, 'final_model_{}_{}.pt'.format(timestamp, epoch)))


# training function
def train_one_epoch(dataloader, model, optimizer, loss_fn, log_path, device):
    running_loss = 0.
    last_loss = 0.

    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(torch.device(device)), y.to(torch.device(device))
        X = X.float()

        optimizer.zero_grad()
        outputs = model(X)

        loss = loss_fn(outputs, y.float())
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        if (batch + 1) % 100 == len(dataloader) % 100:
            last_loss = running_loss / (batch + 1)  # loss per batch
            print('  sample {} loss: {}'.format(batch + 1, last_loss))

    with open(log_path, 'a') as log:
        log.write(f'{last_loss}\n')

    return last_loss
