from code import interact
from doctest import OutputChecker
from fnmatch import fnmatch
from secrets import randbelow
from telnetlib import OUTMRK
from xml.dom.expatbuilder import InternalSubsetExtractor
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor


import h5py as h5
import os
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm
import torch.nn.functional as F
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, create_decoders
from pytorch3dunet.unet3d.utils import number_of_features_per_level, get_class
from pytorch3dunet.unet3d.losses import BCEDiceLoss

import time
import psutil
import sys
#creation of annotation file (done ones for a dataset)

# annotations_file = []
# for filename in os.listdir('C:/Users/andru/Documents/Jupiter_env/ISP_project/AlAu_data/raz_2500ex/3d_structures/'):
#     annotations_file.append(filename)
# annotations_file = pd.DataFrame(annotations_file, columns=['0'])
# print(annotations_file.columns)
# annotations_file.to_csv('C:/Users/andru/Documents/Jupiter_env/ISP_project/AlAu_data/annotations_file.csv')

# https://towardsdatascience.com/how-to-boost-pytorch-dataset-using-memory-mapped-files-6893bff27b99


label_dir = '/trinity/home/andrei.kalinichenko/AlAu_training/perl_2500ex/3d_structures'
img_dir = '/trinity/home/andrei.kalinichenko/AlAu_training/raz_2500ex/3d_structures'
out_dir = '/trinity/home/andrei.kalinichenko/results/loss_adj/L1'
LOG_PATH = '/trinity/home/andrei.kalinichenko/results/loss_adj/L1/test.txt'

learning_rate = 1e-3
# val_percent = 0.1
# weight_decay = 1e-8
# momentum = 0.999
# gradient_clipping = 1.0
# amp = False

# for py pc tests dirs
# label_dir = 'C:/Users/andru/Documents/Jupiter_env/ISP_project/AlAu_data/perl_2500ex/3d_structures'
# img_dir = 'C:/Users/andru/Documents/Jupiter_env/ISP_project/AlAu_data/raz_2500ex/3d_structures'
# out_dir = 'C:/Users/andru/Documents/Jupiter_env/ISP_project/AlAu_data/raz_2500ex'

# dataset loading
dataset_input_storage = []
dataset_label_storage = []
for counter, (img, label) in tqdm(enumerate(zip(os.listdir(img_dir), os.listdir(label_dir)))):
    with np.load(os.path.join(img_dir,img)) as img:
        whole_image = np.expand_dims(img['arr_0'], axis=0)
        for step in range(3):
            X = whole_image[:,0+16*step:32+16*step,0+32*step:64+32*step,0+32*step:64+32*step]
            dataset_input_storage.append(X)
    with np.load(os.path.join(label_dir, label)) as label:
        whole_image = np.expand_dims(label['arr_0'], axis=0)
        for step in range(3):
            X = whole_image[:,0+16*step:32+16*step,0+32*step:64+32*step,0+32*step:64+32*step]
            dataset_label_storage.append(X)
    if counter == 1499: break

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


# dataset = CustomImageDataset(annotations_file = '/trinity/home/andrei.kalinichenko/AlAu_training/annotations_file.csv',
#                                    img_dir = '/trinity/home/andrei.kalinichenko/AlAu_training/perl_2500ex/3d_structures',
#                                    label_dir= '/trinity/home/andrei.kalinichenko/AlAu_training/raz_2500ex/3d_structures')
generator = torch.Generator()
generator.manual_seed(0)
TRAIN_PART = 0.8

dataset = CustomImageDataset(dataset_input_storage, dataset_label_storage)
train_part = round(len(dataset_input_storage)*TRAIN_PART)
train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_part,len(dataset_input_storage) - train_part],
                                                            generator=generator)
print(f'Dataset {len(dataset_input_storage)}\nTrain set : {train_part} images\nValidation set : {len(dataset_input_storage) - train_part} images')

BATCH_SIZE = 10
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(len(train_dataloader))

# memory tests
print(sys.version)
print(psutil.cpu_percent())
print(psutil.virtual_memory()) # physical memory usage 
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0] / 2. ** 30 # memory use in GB...I think 
print(f'memory GB:{memoryUse}')

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))



class Abstract3DUNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=10, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = None
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class UNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

model = UNet3D(1,1).to(device)
# model_dict_PATH = '/trinity/home/andrei.kalinichenko/results/loss_adj/MNE/epoch 51-100/final_model_20230215_115108_101.pt'
# model.load_state_dict(torch.load(model_dict_PATH))
# model.eval()
print(model)

loss_fn = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training function
def train_one_epoch(dataloader, model, optimizer, loss_fn, epoch_index):
    running_loss = 0.
    last_loss = 0.
    running_IoU = [0.,0.]

    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to( torch.device(device)), y.to( torch.device(device))
        X = X.float()

        optimizer.zero_grad()
        outputs = model(X)
        
        loss = loss_fn(outputs, y.float())
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

        for n, i in enumerate(mean_iou(outputs.to('cpu').detach().numpy(), y.to('cpu').detach().numpy())):
            running_IoU[n] += i

        if (batch+1) % 100 == len(dataloader) % 100:
            last_loss = running_loss / (batch + 1) # loss per batch
            print('  sample {} loss: {}'.format(batch + 1, last_loss))
            last_IoU_1 = running_IoU[0] / (batch + 1) # loss per batch
            last_IoU_2 = running_IoU[1] / (batch + 1) # loss per batch
    with open(LOG_PATH, 'a') as log:
        log.write(f'{last_loss},{last_IoU_1},{last_IoU_2}\n')
    print('  sample {} IoU 1: {}, 2: {}'.format(batch + 1, last_IoU_1, last_IoU_2))

    return last_loss

def mean_iou(pred, y):

    segment = np.round(pred)
    segment[segment > 2] = 2
    for mark in range(1,3):
        pred_mask = np.zeros_like(segment)
        pred_mask[segment == mark] = 1
        y_mask = np.zeros_like(segment)
        y_mask[y == mark] = 1
        intersection = np.logical_and(y_mask, pred_mask)
        union = np.logical_or(y_mask, pred_mask)
        yield round((np.sum(intersection) / (np.sum(union) + 1e-10)), 6)

# logger inicialisation
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter(os.path.join(out_dir, 'AuAi_trainer_{}'.format(timestamp)))
epoch_number = 0

EPOCHS = 100

best_tloss = 1_000_000.

# main train/val cycle
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(train_dataloader, model, optimizer, loss_fn, epoch_index=epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_IoU = [0.,0.]
    running_tloss = 0.
    for batch, (tinputs, tlabels) in tqdm(enumerate(test_dataloader)):
        tinputs, tlabels = tinputs.to(torch.device(device)), tlabels.to(torch.device(device))
        tinputs = tinputs.float()
        with torch.no_grad():
            toutputs = model(tinputs)
            
        tloss = loss_fn(toutputs, tlabels.float())
        running_tloss += tloss
        for n, i in enumerate(mean_iou(toutputs.to('cpu').detach().numpy(), tlabels.to('cpu').detach().numpy())):
            running_IoU[n] += i
    avg_tloss = running_tloss/(batch+1)
    avg_IoU_1 = running_IoU[0]/(batch+1)/2
    avg_IoU_2 = running_IoU[1]/(batch+1)/2
    print('LOSS train {} test {}'.format(avg_loss, avg_tloss))
    print('IoU test 1: {}, 2: {}'.format(avg_IoU_1, avg_IoU_2))
    with open(LOG_PATH, 'a') as log:
        log.write(f'{avg_tloss},{avg_IoU_1},{avg_IoU_2}\n')
    # Track best performance, and save the model's state
    # if avg_tloss < best_tloss:
    #     best_tloss = avg_tloss
    #     model_path = os.path.join(out_dir, 'model_{}_{}'.format(timestamp, epoch_number))
    #     torch.save(model.state_dict(), model_path)

    epoch_number += 1
    if epoch_number%10 == 9:
        torch.save(model.state_dict(), os.path.join(out_dir, 'final_model_{}_{}.pt'.format(timestamp, epoch_number)))


torch.save(model.state_dict(), os.path.join(out_dir, 'final_model_{}_{}.pt'.format(timestamp, epoch_number)))
