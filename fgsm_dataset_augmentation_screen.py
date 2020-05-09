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
batch_size = 128
im_height = 64
im_width = 64
num_epochs = 1

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
])

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=data_dir / 'train', transform=data_transforms),
    'valid': datasets.ImageFolder(root=data_dir / 'val', transform=data_transforms),
    'test': datasets.ImageFolder(root=data_dir / 'test', transform=data_transforms)
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

# Create iterators for the Data loaded using DataLoader module
train_data_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")



import resnet_modified
model = resnet_modified.resnet152(pretrained=False, decay_factor=0.04278)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# best_model_path = "./models/resnet152_best_model_epoch_34.pth"
best_model_path = "./models/resnet152_best_model_state_dict_v2_50.pth"
model.load_state_dict(torch.load(best_model_path))




# FGSM attack code, from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image



import cv2
def save_fgsm(model, criterion, optimizer, scheduler, dataloaders, epsilon, idx_to_class_map, save_location,
              num_epochs=1, verbose=True, save=True):
    since = time.time()

    tr_acc, val_acc = [], []
    tr_loss, val_loss  = [], []
    
    adv_examples = []
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), end=": ")
        
        # Each epoch has a training and validation phase
        phase = 'val'
        model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs.requires_grad = True

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            init_pred = outputs.max(1, keepdim=True)[1] # get the index of the max log-probability

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = inputs.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(inputs, epsilon, data_grad)

            # Re-classify the perturbed image
            outputs = model(perturbed_data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            final_pred = outputs.max(1, keepdim=True)[1] # get the index of the max log-probability
            ex = inputs.squeeze().detach().cpu().numpy()
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            
            for i in range(batch_size):
                file_name = save_location + str(idx_to_class_map[int(labels[0].detach().cpu().numpy())]) + "/images/" + str(np.random.randint(0, 1e15)) + ".JPEG"
                
                img = np.transpose(adv_ex[i] * 4096, (1, 2, 0))
                img = Image.fromarray(img.astype(np.uint8))
                img.save(file_name)
#                 img = cv2.imread(file_name)
                denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 7, 21)
                print(file_name)
                cv2.imwrite(file_name, denoised_img)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if i % 100 == 0:
                time_elapsed = time.time() - since
                print('Time Elapsed: {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
            
            break

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        if phase == 'train':
            tr_acc.append(epoch_acc)
            tr_loss.append(epoch_loss)
        elif phase == 'val':
            val_acc.append(epoch_acc)
            val_loss.append(epoch_loss)
        
        if verbose:        
            print('Validation Loss: {:.4f}, Acc: {:.4f}'.format(
                val_loss[-1], val_acc[-1]))

    return val_acc[0], val_loss[0], adv_examples



import shutil
import os

# List of TinyImageNet classes
tiny_imagenet_classes = os.listdir("./data/tiny-imagenet-200/train/")

idx_to_class_map = {}
class_to_name_map = {}
with open('./data/tiny-imagenet-200/words.txt', "r") as f:
    i = 0
    for line in f:
        one_line = line.split()
        if one_line[0] in tiny_imagenet_classes:
            idx_to_class_map[i] = one_line[0]
            class_name = " ".join(one_line[1:]).split(',')[0]
            class_to_name_map[one_line[0]] = class_name
            i += 1

            
            
            
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


            
epsilons = [5e-5]

lr = 2.61e-5
loss_func = nn.CrossEntropyLoss().cuda(device)
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# optimizer_ft = optim.Adam(model.parameters(), lr=lr)

# "./data/tiny-imagenet-200/train/n12267677/images/n12267677_139.JPEG"
image_location = "./data/tiny-imagenet-200/train_adversarial/"

# Run test for each epsilon
acc, loss, ex = save_fgsm(model, loss_func, optimizer_ft, None, dataloaders, epsilons[0], 
                          idx_to_class_map, image_location, verbose=False)

print("Done")