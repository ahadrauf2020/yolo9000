from model.residual_attention_network import ResidualAttentionModel_92_32input_my_update as ResidualAttentionModel
# from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    model = ResidualAttentionModel()
    model = torch.nn.DataParallel(model)
        
    optimizer = torch.optim.SGD(model.parameters(), 0.001, \
                                momentum=0.9, \
                                weight_decay=1e-4)
    checkpoint = torch.load('best_model/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()


if __name__ == '__main__':
    main()


