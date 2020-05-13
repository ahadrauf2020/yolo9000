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
from Residual_Attention_Network.model.residual_attention_network import ResidualAttentionModel_92_32input_my_update as ResidualAttentionModel
import resnet_modified
from Ensemble import Ensemble


num_classes = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    



class EnsembleTable():
    def __init__(self, fgsm_dataloader=None, blurred_dataloader=None):
        image_datasets, normal_dataloaders, dataset_sizes, class_names = self.load_data()
        
        self.dataloaders = normal_dataloaders
        self.fgsm_dataloader = normal_dataloaders if fgsm_dataloader is None else fgsm_dataloader
        self.blurred_dataloader = normal_dataloaders if blurred_dataloader is None else blurred_dataloader
        self.models = self.load_models()

    def load_data(self, batch_size=500):
        data_dir = './data/tiny-imagenet-200'
        im_height = 64
        im_width = 64
        phases = ['train', 'val', 'test']    

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
        print("Loaded normal data of size", dataset_sizes)
        return image_datasets, dataloaders, dataset_sizes, class_names


    # Load Models
    def load_models(self):
        resnet_model = resnet_modified.resnet152(pretrained=False, decay_factor=0.04278)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, num_classes)
        best_model_path = "./models/resnet152_best_model_state_dict.pth"
        resnet_model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
        resnet_model = resnet_model.to(device)

        vgg_model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
        num_ftrs = vgg_model.classifier[6].in_features
        vgg_model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        vgg_model.load_state_dict(torch.load('./models/vgg19_bn_best_model.pth', map_location=torch.device(device)))
        vgg_model = vgg_model.to(device)

        dense_model = torchvision.models.densenet169(pretrained=True)
        num_ftrs = dense_model.classifier.in_features
        dense_model.classifier = nn.Linear(num_ftrs, num_classes)
        dense_model.load_state_dict(torch.load('./models/densenet169_best_model_state_dict_v2_65.pth', map_location=torch.device(device)))
        dense_model = dense_model.to(device)

    #     attention_model = ResidualAttentionModel()
    #     attention_model =  torch.nn.DataParallel(attention_model)
    #     checkpoint = torch.load('./models/chris_resnet_model_best.pth.tar', map_location=torch.device(device))
    #     state_dict =checkpoint['state_dict']
    #     attention_model.load_state_dict(state_dict,False)
    #     attention_model = attention_model.to(device)    

        return [resnet_model, vgg_model, dense_model]



    def print_table(self):
        model_names = ["Resnet152", "VGG19_bn", "DenseNet", "ResAttNet"]
            
        print("Validation Accuracy Table")
        for i in range(len(self.models)):
            criterion = nn.CrossEntropyLoss()
            ensemble_solver = Ensemble([self.models[i]])
            top1_acc, top5_acc, val_loss = ensemble_solver.evaluate_all(criterion, self.dataloaders)
            fgsm_top1_acc, fgsm_top5_acc, fgsm_val_loss = ensemble_solver.evaluate_all(criterion, self.fgsm_dataloader)
            blurred_top1_acc, blurred_top5_acc, blurred_val_loss = ensemble_solver.evaluate_all(criterion, self.blurred_dataloader)
            print("{} = top1_acc: {}, top5_acc:{}, fgsm_top1_acc:{}, blurred_top1_acc:{}".format(model_names[i], top1_acc, top5_acc, fgsm_top1_acc, blurred_top1_acc))
            
            
        resnet_model, vgg_model, dense_model = self.models[0], self.models[1], self.models[2]
        
        combo = [
#             [resnet_model, dense_model, vgg_model, attention_model],
#             [resnet_model, dense_model, attention_model],
#             [resnet_model, vgg_model, attention_model],
            [resnet_model, dense_model, vgg_model]
        ]
        combo_names = [
#             ["Resnet152, VGG19_bn, DenseNet, ResAttNet"],
#             ["Resnet152, DenseNet, ResAttNet"],
#             ["Resnet152, VGG19_bn, ResAttNet"],
            ["Resnet152, VGG19_bn, DenseNet"]
        ]
            
        # Ensemble by Averaging logits
        for i in range(len(combo)):
            criterion = nn.CrossEntropyLoss()
            ensemble_solver = Ensemble(combo[i])
            top1_acc, top5_acc, val_loss = ensemble_solver.evaluate_all(criterion, self.dataloaders)
            fgsm_top1_acc, fgsm_top5_acc, fgsm_val_loss = ensemble_solver.evaluate_all(criterion, self.fgsm_dataloader)
            blurred_top1_acc, blurred_top5_acc, blurred_val_loss = ensemble_solver.evaluate_all(criterion, self.blurred_dataloader)
            print(combo_names[i])
            print("Validation top1_acc: {}, top5_acc:{}, fgsm_top1_acc:{}, blurred_top1_acc:{}".format(top1_acc, top5_acc, fgsm_top1_acc, blurred_top1_acc))

         # Ensemble by Majority Vote
        for i in range(len(combo)):
            criterion = nn.CrossEntropyLoss()
            ensemble_solver = Ensemble(combo[i])
            top1_acc, top5_acc, val_loss = ensemble_solver.evaluate_all(criterion, self.dataloaders, "maj vote")
            fgsm_top1_acc, fgsm_top5_acc, fgsm_val_loss = ensemble_solver.evaluate_all(criterion, self.fgsm_dataloader, "maj vote")
            blurred_top1_acc, blurred_top5_acc, blurred_val_loss = ensemble_solver.evaluate_all(criterion, self.blurred_dataloader, "maj vote")
            print(combo_names[i])
            print("Validation top1_acc: {}, top5_acc:{}, fgsm_top1_acc:{}, blurred_top1_acc:{}".format(top1_acc, top5_acc, fgsm_top1_acc, blurred_top1_acc))
