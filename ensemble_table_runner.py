import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from EnsembleTable import EnsembleTable

paths = {
    'resnet152': "./models/resnet152_best_model_state_dict.pth",
    'vgg19_bn': './models/vgg19_bn_best_model.pth',
    'dense169': './models/densenet169_best_model_state_dict_v2_65.pth',
    'resatt': './models/res_att_best_model_epoch_15.pth'
}

# Load the blurred Data
blurred_data_dir = './adversarial_data/val_adversarial/'
num_classes = 200
im_height = 64
im_width = 64
phases = ['train', 'val', 'test']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
batch_size = 500

def load_blurred_data(batch_size=500):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    image_datasets = {
        'val': datasets.ImageFolder(blurred_data_dir, transform=data_transforms)
    }
    dataloaders = {
          'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
    }
    dataset_sizes = {'val': len(image_datasets['val'])}
    class_names = image_datasets['val'].classes
    print("number of classes", len(class_names))
    return dataloaders, dataset_sizes

blurred_dataloaders, blurred_dataset_sizes = load_blurred_data()
print("blurred dataset size", blurred_dataset_sizes)


table = EnsembleTable(paths=paths, fgsm_dataloader=None, blurred_dataloader=blurred_dataloaders, fgsm_dataset_sizes=None, blurred_dataset_sizes=blurred_dataset_sizes)
table.print_table()
