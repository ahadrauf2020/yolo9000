import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import time
from torchsummary import summary
from torch.optim import lr_scheduler
import copy

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from collections import OrderedDict
import shutil 
# reference : https://github.com/automan000/CyclicLR_Scheduler_PyTorch
from CyclicLR_Scheduler_PyTorch.cyclic_lr_scheduler import CyclicLR
from Residual_Attention_Network.model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
import resnet_modified


# Load the Data
data_dir = './data/tiny-imagenet-200'
num_classes = 200


# Create the training data generator
batch_size = 500
im_height = 64
im_width = 64
phases = ['train', 'val', 'test']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
batch_size = 500

def load_data(batch_size=500):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    

    # Load Data from folders
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)
    }

    # subset_indices = np.random.permutation(range(100))
    # dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, 
    #                              sampler=SubsetRandomSampler(subset_indices)) for x in phases}

    dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
                  'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
                  'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)}
    dataset_sizes = {x: len(image_datasets[x]) for x in phases}
    class_names = image_datasets['train'].classes
    return image_datasets, dataloaders, dataset_sizes, class_names

# image_datasets, dataloaders, dataset_sizes, class_names = load_data()


class Ensemble():
    def __init__(self, models):
        self.models = models
        self.loss = 0.0
        self.top5_acc = 0.0
        self.top1_acc = 0.0
        
    def get_num_corrects(self, output, target, topk=(1,)):
        res = []
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
        return res
        
    def find_majority_vote(self, preds):
        maj_vote = torch.zeros(preds.shape[1])
        for i in range(preds.shape[1]):
            _, counts = np.unique(preds[:, i], return_counts=True)
            maj_vote[i] = preds[np.argmax(counts), i]
        maj_vote = maj_vote.to(device)
        return maj_vote
    
    def evaluate_testdata(self, inputs, mode='average'):
        inputs = inputs.to(device)
        phase = 'val'
        for m in self.models:
            m.eval()
        with torch.no_grad():
            if mode == 'average':
                # Take average of the output to make prediction
                outputs = torch.zeros(1, num_classes)
                for m in self.models:
                    outputs += m(inputs)
                outputs /= len(self.models)
                _, preds = torch.max(outputs, 1)
            else:
                # Majority vote
                loss = 0
                predictions = torch.zeros(len(self.models), inputs.shape[0])
                for i in range(len(self.models)):
                    outputs = self.models[i](inputs)
                    _, preds = torch.max(outputs, 1)
                    predictions[i, :] = preds
                    loss += criterion(outputs, labels)
                preds = self.find_majority_vote(predictions)
        return preds

                
    def evaluate_all(self, criterion, mode='average'):
        running_loss = 0.0
        running_corrects = 0
        running_corrects1 = 0
        running_corrects5 = 0
        phase = 'val'
        for m in self.models:
            m.eval()
            
        with torch.no_grad():
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
#                 if isinstance(m, ResidualAttentionModel):
#                 inputs = torch.nn.functional.interpolate(inputs, (32, 32))
                if mode == 'average':
                    # Take average of the output to make prediction
                    outputs = torch.zeros(batch_size, num_classes)
                    for m in self.models:
                        outputs += m(inputs)
                    outputs /= len(self.models)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                else:
                    # Majority vote
                    loss = 0
                    predictions = torch.zeros(len(self.models), inputs.shape[0])
                    for i in range(len(self.models)):
                        outputs = self.models[i](inputs)
                        _, preds = torch.max(outputs, 1)
                        predictions[i, :] = preds
                        loss += criterion(outputs, labels)
                    preds = self.find_majority_vote(predictions)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                corr1, corr5 = self.get_num_corrects(outputs, labels, topk=(1, 5))
                running_corrects1 += corr1[0]
                running_corrects5 += corr5[0]
                
            self.loss = running_loss / dataset_sizes[phase]
            self.top1_acc = running_corrects1.double() / dataset_sizes[phase]
            self.top5_acc = running_corrects5.double() / dataset_sizes[phase]
        return self.top1_acc, self.top5_acc, self.loss
