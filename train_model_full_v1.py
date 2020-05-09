import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import time
from torchsummary import summary
from torch.optim import lr_scheduler
import copy

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from collections import OrderedDict
import shutil


# Load the Data

# Load the Data
data_dir = './data/tiny-imagenet-200'
num_classes = 200

# Create the training data generator
batch_size = 500
im_height = 64
im_width = 64

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

phases = ['train', 'val', 'test']

dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
              'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
              'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)}
dataset_sizes = {x: len(image_datasets[x]) for x in phases}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import resnet_modified
model = resnet_modified.resnet152(pretrained=False, decay_factor=0.04278)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
best_model_path = "./models/resnet152_best_model_state_dict.pth"
model.load_state_dict(torch.load(best_model_path))


def plot_result(x_scale, tr, val, title, y_label, ax=plt):
    ax.set_title(title)
    if title == 'loss':
        ax.plot(x_scale, tr, label='training loss')
        ax.plot(x_scale, val, label='validation loss')
    else:
        ax.plot(x_scale, tr, label='training accuracy')
        ax.plot(x_scale, val, label='validation accuracy')
    ax.set_xlabel("Epochs")
    ax.set_ylabel(y_label)
    ax.legend()

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10, verbose=True, save=True, start_count=0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    tr_acc, val_acc = [], []
    tr_loss, val_loss  = [], []
    
    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), end=": ", flush=True)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                tr_acc.append(epoch_acc)
                tr_loss.append(epoch_loss)
            elif phase == 'val':
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        if verbose:
            time_elapsed = time.time() - since
            print('Time Elapsed {:.0f}m {:.0f}s --> Training Loss: {:.4f}, Acc: {:.4f}; Validation Loss: {:.4f}, Acc: {:.4f}'.format(
                time_elapsed // 60, time_elapsed % 60, tr_loss[-1], tr_acc[-1], val_loss[-1], val_acc[-1]), flush=True)
        if save:
            torch.save(model.state_dict(), './models/resnet152_best_model_epoch_' + str(epoch + start_count) + '.pth')
            np.save('full_training_in_progress_part2.npy', np.array([tr_acc, val_acc, tr_loss, val_loss]))

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, tr_acc, val_acc, tr_loss, val_loss





lr = 2.61e-5
decay_factor = 0.04278
batch_size = 128

print("learning rate: {}, decay_factor: {}, batch_size: {}".format(lr, decay_factor, batch_size))

# criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

# dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
#           'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True)}

# model, tr_acc, val_acc, tr_loss, val_loss = train_model(model, criterion, optimizer_ft, scheduler, dataloaders, 
#                                                         num_epochs=15, verbose=True)
    
# torch.save(model.state_dict(), './models/resnet152_best_model_state_dict_v2_15.pth')
# np.save('full_training_15.npy', np.array([tr_acc, val_acc, tr_loss, val_loss]))





data_transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
])

# Load Data from folders
# image_datasets = {
#     'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms_train),
#     'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms),
#     'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)
# }

# phases = ['train', 'val', 'test']

# dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
#               'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
#               'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)}


# criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

# model, tr_acc2, val_acc2, tr_loss2, val_loss2 = train_model(model, criterion, optimizer_ft, scheduler, dataloaders, 
#                                                         num_epochs=35, verbose=True)
    
# torch.save(model.state_dict(), './models/resnet152_best_model_state_dict_v2_50.pth')
# np.save('full_training_50.npy', np.array([tr_acc2, val_acc2, tr_loss2, val_loss2]))
# tr_acc.extend(tr_acc2)
# val_acc.extend(val_acc2)
# tr_loss.extend(tr_loss2)
# val_loss.extend(val_loss2)
# np.save('full_training_50_v2.npy', np.array([tr_acc, val_acc, tr_loss, val_loss]))


image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms_train),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)
}

phases = ['train', 'val', 'test']

dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
              'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
              'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)}


criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

model, tr_acc2, val_acc2, tr_loss2, val_loss2 = train_model(model, criterion, optimizer_ft, scheduler, dataloaders, 
                                                        num_epochs=15, verbose=True)
    
torch.save(model.state_dict(), './models/resnet152_best_model_state_dict_v2_65.pth')
np.save('full_training_65.npy', np.array([tr_acc2, val_acc2, tr_loss2, val_loss2]))






# fig, (ax1, ax2) = plt.subplots(1, 2)
# plot_result(range(10), tr_acc, val_acc, 'acc', 'Accuracy', ax=ax1)
# plot_result(range(10), tr_loss, val_loss, 'loss', 'Loss', ax=ax2)
# fig.tight_layout()
# plt.show()


np.save("train_data_resnet50_fh_v1.npy", np.array(train_data))
np.save("val_data_resnet50_fh_v1.npy", np.array(val_data))
