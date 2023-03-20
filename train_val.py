import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime

import os
import random

from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm
from models import UNet3D, VNet

import h5py

import psutil
import sys
from CustomImageDataset import h5_dataset
from predict_set import predict_set

OPTIMIZERS_LIST = ('Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'SGD', 'RAdam', 'Rprop',
                    'RMSprop', 'NAdam', 'LBFGS',)
LOSSES_LIST = ("CrossEntropyLoss", "BCELoss", "MSELoss", "L1Loss", "SmoothL1Loss", "KLDivLoss", "CosineEmbeddingLoss",
                "TripletMarginLoss", "HingeEmbeddingLoss", "MultiMarginLoss", )
DATE_FORMAT = '%Y-%m-%d_%H-%m-%S'



def start_training(h5_file_path, gamma, out_dir, log_path, train_part, model, loss_fn, learning_rate, epochs, batch_size, optimizer, debug_launch):
    



    generator = torch.Generator()
    generator.manual_seed(0)

    dataset = h5_dataset(h5_file_path)

    if debug_launch:
        train_part = 100
        test_part = 50
        leftover = len(dataset) - train_part - test_part
        train_dataset, test_dataset, _ = torch.utils.data.random_split(dataset,
                                                                    [train_part, test_part, leftover],
                                                                    generator=generator)

    else:
        train_part = round(len(dataset) * train_part)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                    [train_part, len(dataset) - train_part],
                                                                    generator=generator)
    print(
        f'Dataset {len(dataset)}\nTrain set : {train_part} images\nValidation set : {len(dataset) - train_part} images')

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

    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # logger inicialisation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # main train/val cycle
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(train_dataloader, model, optimizer, loss_fn, log_path, device)

        scheduler.step()
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

    model_name = os.path.join(out_dir, 'final_model_{}_{}.pt'.format(timestamp, epoch))
    torch.save(model.state_dict(), model_name)


    predict_set(train_dataloader, model_name, label, out_dir, set_size=set_size)
    predict_set(test_dataloader, model_name, label, out_dir, set_size=set_size)


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
