import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image


# Load the Data

data_dir = pathlib.Path('./data/tiny-imagenet-200')
image_count = len(list(data_dir.glob('**/*.JPEG')))
CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
num_classes = len(CLASS_NAMES)
print('Discovered {} images in {} classes'.format(image_count, num_classes))

# Create the training data generator
batch_size = 32
im_height = 64
im_width = 64
num_epochs = 1

data_transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=data_dir / 'train', transform=data_transforms_train),
    'valid': datasets.ImageFolder(root=data_dir / 'val', transform=data_transforms_test),
    'test': datasets.ImageFolder(root=data_dir / 'test', transform=data_transforms_test)
}

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")





import resnet_modified
model2 = resnet_modified.resnet50(pretrained=True, decay_factor=0.1)


# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = model2.fc.in_features

model2.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes), # Since 10 possible outputs
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)


# Convert model to be used on GPU
model2 = model2.to('cuda:0')

model2.eval()



import train_model_within_programs
args = ["./data/tiny-imagenet-200/", "--epochs", "3", "--filename", "resnet50_fh_v1", "--print-freq", "50"]
train_data, val_data = train_model_within_programs.run(args, model2, data['train'], data['valid'])


np.save("train_data_resnet50_fh_v1.npy", np.array(train_data))
np.save("val_data_resnet50_fh_v1.npy", np.array(val_data))