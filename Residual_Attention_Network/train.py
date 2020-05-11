# from __future__ import print_function, division
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import torchvision
# from torchvision import transforms, datasets, models
# import os
# import cv2
# import time
# import sys
# from collections import OrderedDict
# import shutil 
# # from model.residual_attention_network_pre import ResidualAttentionModel
# # based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

# model_file = 'model_92_sgd.pkl'


# # for test
# def test(model, val_loader, btrain=False, model_file='model_92_sgd.pkl'):
#     # Test
#     if not btrain:
#         model.load_state_dict(torch.load(model_file))
#     model.eval()

#     correct = 0.000001
#     total = 0.000001
#     #
#     class_correct = list(0. for i in range(200))
#     class_total = list(0. for i in range(200))

#     for images, labels in val_loader:
#         # images = Variable(images.cuda())
#         # labels = Variable(labels.cuda())

#         # print("CCCCC", images.size())
#         images = images.resize_((64, 3, 32, 32))
        
#         outputs = model(images)
#         # print('outputs', outputs.size())
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         print('predicted', predicted)
#         print('labels', labels)
#         if labels.size(0) != predicted.size(0):
#             continue
#         correct += (predicted == labels.data).sum()
        
#         c = (predicted == labels.data).squeeze()
#         for i in range(20):
#             label = labels.data[i]
#             class_correct[label] += c[i]
#             class_total[label] += 1
#         break
#     print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
#     print('Accuracy of the model on the test images:', float(correct)/total)
#     for i in range(200):
#         print('Accuracy of %5s : %2d %%' % (
#             classes[i], 100 * class_correct[i] / class_total[i]))
#     return correct / total


# # Image Preprocessing
# # transform = transforms.Compose([
# #     transforms.RandomHorizontalFlip(),
# #     transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
# #     # transforms.Scale(224),
# #     transforms.ToTensor()
# # ])
# # test_transform = transforms.Compose([
# #     transforms.ToTensor()
# # ])

# # when image is rgb, totensor do the division 255
# # CIFAR-10 Dataset
# # train_dataset = datasets.CIFAR10(root='./data/',
# #                                train=True,
# #                                transform=transform,
# #                                download=True)

# # test_dataset = datasets.CIFAR10(root='./data/',
# #                               train=False,
# #                               transform=test_transform)

# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
#     # transforms.Scale(224),
#     transforms.ToTensor()
# ])

# test_transform = transforms.Compose([
#     transforms.ToTensor()
# ])


# datadir = sys.argv[1]
# traindir = os.path.join(datadir, 'train')
# # testdir = os.path.join(datadir, 'test')
# valdir = os.path.join(datadir, 'val')

# train_dataset = datasets.ImageFolder(traindir, transform=transform)

# # test_dataset = datasets.ImageFolder(testdir, transform=test_transform)
# val_dataset = datasets.ImageFolder(valdir, transform=test_transform)

# # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(train_dataset, 
#                                         batch_size=64, 
#                                         shuffle=True,
#                                         num_workers=8
#                                         )


# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                           batch_size=64,
#                                           shuffle=False)


# # classes = ('plane', 'car', 'bird', 'cat',
# #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# classes = ('n02795169', 'n02769748', 'n07920052', 'n02917067', 'n01629819', 'n02058221', 
#     'n02793495', 'n04251144', 'n02814533', 'n02837789', 'n01770393', 'n01910747', 'n03649909', 
#     'n02124075', 'n01774750', 'n06596364', 'n03838899', 'n02480495', 'n09256479', 'n03085013', 
#     'n01443537', 'n04376876', 'n03404251', 'n03930313', 'n03089624', 'n04371430', 'n04254777', 
#     'n02909870', 'n07614500', 'n02977058', 'n04259630', 'n07579787', 'n02950826', 'n02279972', 
#     'n03424325', 'n03854065', 'n02403003', 'n01742172', 'n01882714', 'n03977966', 'n02669723', 
#     'n02226429', 'n04366367', 'n02002724', 'n03891332', 'n01768244', 'n02509815', 'n03544143', 
#     'n02321529', 'n02099601', 'n02948072', 'n04456115', 'n02236044', 'n03126707', 'n02074367', 
#     'n03255030', 'n01950731', 'n02268443', 'n04501370', 'n03970156', 'n04099969', 'n04023962', 
#     'n02085620', 'n02823428', 'n04265275', 'n02113799', 'n01784675', 'n03706229', 'n03100240', 
#     'n04532106', 'n02788148', 'n07753592', 'n03983396', 'n04399382', 'n03902125', 'n02814860', 
#     'n03014705', 'n09428293', 'n02481823', 'n04597913', 'n01944390', 'n03355925', 'n07871810', 
#     'n03042490', 'n02190166', 'n04486054', 'n04008634', 'n02906734', 'n02699494', 'n04070727', 
#     'n01855672', 'n09246464', 'n02364673', 'n07768694', 'n02883205', 'n04532670', 'n02815834', 
#     'n02165456', 'n04540053', 'n02802426', 'n04356056', 'n03670208', 'n04562935', 'n01641577', 
#     'n07615774', 'n07734744', 'n03584254', 'n01698640', 'n04507155', 'n02125311', 'n03179701', 
#     'n07873807', 'n04179913', 'n04560804', 'n03393912', 'n02841315', 'n02843684', 'n09193705', 
#     'n02437312', 'n04275548', 'n04118538', 'n02099712', 'n07747607', 'n03250847', 'n04133789', 
#     'n02094433', 'n04074963', 'n02129165', 'n03637318', 'n02056570', 'n02410509', 'n03980874', 
#     'n03400231', 'n03814639', 'n03026506', 'n01644900', 'n04398044', 'n02666196', 'n03444034', 
#     'n04487081', 'n02486410', 'n02808440', 'n04149813', 'n12267677', 'n03662601', 'n02233338', 
#     'n07711569', 'n02791270', 'n04465501', 'n03599486', 'n07720875', 'n03447447', 'n03804744', 
#     'n04311004', 'n07695742', 'n07583066', 'n07715103', 'n04328186', 'n01917289', 'n02106662', 
#     'n02927161', 'n02395406', 'n02231487', 'n02123394', 'n03976657', 'n02423022', 'n03770439', 
#     'n04067472', 'n02206856', 'n04285008', 'n03617480', 'n03733131', 'n02415577', 'n04146614', 
#     'n03388043', 'n01945685', 'n02892201', 'n03160309', 'n02281406', 'n02999410', 'n02504458', 
#     'n04596742', 'n02132136', 'n03763968', 'n03796401', 'n07875152', 'n01983481', 'n07749582', 
#     'n01774384', 'n03201208', 'n01984695', 'n02963159', 'n02123045', 'n09332890', 'n03992509', 
#     'n02988304', 'n04417672', 'n02730930', 'n03937543', 'n03837869')
# # model = ResidualAttentionModel().cuda()
# model = ResidualAttentionModel()

# lr = 0.1  # 0.1
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
# is_train = True
# is_pretrain = False
# acc_best = 0
# total_epoch = 10
# if is_train is True:
#     if is_pretrain == True:
#         model.load_state_dict((torch.load(model_file)))
#     # Training
#     for epoch in range(total_epoch):
#         model.train()
#         tims = time.time()

#         for i, (images, labels) in enumerate(train_loader):
#             # images = Variable(images.cuda())
#             # images = Variable(images)

#             # labels = Variable(labels.cuda())
#             # labels = Variable(labels)

#             # Forward + Backward + Optimize
#             optimizer.zero_grad()
#             outputs = model(images)

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             # print("hello")
#             # if (i+1) % 100 == 0:
#             print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.item()))
#             # print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, total_epoch, i+1, len(train_loader), loss.data[0]))
#             # print("Acc so far:", test(model, test_loader, btrain=True))
#             break
#         print('the epoch takes time:',time.time()-tims)
#         print('evaluate validation set:')
#         acc = test(model, val_loader, btrain=True)
#         if acc > acc_best:
#             acc_best = acc
#             print('current best acc,', acc_best)
#             torch.save(model.state_dict(), model_file)
#         # Decaying Learning Rate
#         if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
#             lr /= 10
#             print('reset learning rate to:', lr)
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
#                 print(param_group['lr'])
#             # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#             # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
#     # Save the Model
#     torch.save(model.state_dict(), 'last_model_92_sgd.pkl')

# else:
#     test(model, val_loader, btrain=False)
import argparse
import os
import random
import shutil
import time
import warnings

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

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='./model_92_sgd.pkl',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet152)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():

    args = parser.parse_args()
    print('Epoch:', args.epochs,'batch_size:', args.batch_size)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch]()
        model = ResidualAttentionModel()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((32, 32), padding=4), 
            # transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4),   #left, top, right, bottom
            # transforms.Scale(224),
            transforms.ToTensor()
     
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)


        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

def load():
    model = ResidualAttentionModel()
    model =  torch.nn.DataParallel(model)
        
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=1e-4)
    checkpoint = torch.load('cs182_residual_attention_nn/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    print("Epoch:{}".format(epoch), "top1:", top1, "top5:", top5)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
