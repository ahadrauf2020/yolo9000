import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from Residual_Attention_Network.model.residual_attention_network import ResidualAttentionModel_92_32input_my_update as ResidualAttentionModel
import resnet_modified

num_classes = 200
phases = ['train', 'val', 'test']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
batch_size = 500


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
            unique, counts = np.unique(preds[:, i], return_counts=True)
            max_val = unique[np.argmax(counts)]
            maj_vote[i] = torch.from_numpy(np.array([max_val])).float().to(device)
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
                outputs = torch.zeros(1, num_classes).to(device)
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

                
    def evaluate_all(self, criterion, dataloaders, dataset_sizes, mode='average'):
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
                
                if mode == 'average':
                    # Take average of the output to make prediction
#                     outputs = torch.zeros(batch_size, num_classes).to(device)
                    outputs = None
                    for m in self.models:
                        if outputs is None:
                            outputs = m(inputs)
                        else:
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
                
                if mode == 'average':
                    corr1, corr5 = self.get_num_corrects(outputs, labels, topk=(1, 5))
                    running_corrects1 += corr1[0]
                    running_corrects5 += corr5[0]
                else:
                    running_corrects1 += torch.sum(preds == labels.data)
                
            self.loss = running_loss / dataset_sizes[phase]
            if mode == 'average':
                self.top1_acc = running_corrects1.double() / dataset_sizes[phase]
                self.top5_acc = running_corrects5.double() / dataset_sizes[phase]
            else:
                self.top1_acc = running_corrects1.double() / dataset_sizes[phase]
        return self.top1_acc, self.top5_acc, self.loss
