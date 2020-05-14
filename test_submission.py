import sys
import pathlib
from PIL import Image
import numpy as np
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import resnet_modified
from Residual_Attention_Network.model.residual_attention_network import ResidualAttentionModel_92_32input_my_update as ResidualAttentionModel
from Ensemble import Ensemble


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
num_classes = 200


def load_models():
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


def main():
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])
    im_height, im_width = 64, 64

    models = load_models()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255))))
    ])
    
    evalcsv_path = sys.argv[1]
    print("path to the eval.csv file:", evalcsv_path)

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
#         eval_dir = pathlib.Path('eval.csv')
        eval_dir = pathlib.Path(evalcsv_path)
        for line in eval_dir.open():
        # for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            ensemble_solver = Ensemble(models)
            predicted = ensemble_solver.evaluate_testdata(img)
            print("predicted class:", CLASSES[predicted])
            print()

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()
