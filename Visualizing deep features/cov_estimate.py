import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import resnet
from ISDA_imagenet import ISDALoss
import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_url', default='./', type=str, help='IMAGENET')
parser.add_argument('--train_url', default='./', type=str, help='Path to model and saved Cov Matrix (default: ./)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--lambda_0', default=7.5, type=float, help='The hyper-parameter \lambda_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
                         'We adopt 1 for DenseNets and 7.5 for ResNets and ResNeXts, except for using 5 for ResNet-101.')


def main():
    args = parser.parse_args()
    print(args)

    print('=========> Load checkpoint from train_url')
    checkpoint_dir = os.path.join(args.train_url, 'resnet50-19c8e357.pth')
    print('checkpoint_dir:', checkpoint_dir)

    checkpoint = torch.load(checkpoint_dir)
    model = resnet.resnet50()
    model.load_state_dict(checkpoint)
    feature_num = model.feature_num
    print('Number of final features: {}'.format(int(model.feature_num)))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()
    criterion_isda = ISDALoss(feature_num, 1000).cuda()
    criterion_ce = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    print('=========> Load Dataset')
    traindir = os.path.join(args.data_url, 'train')
    valdir = os.path.join(args.data_url, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    print('=========> Start Training')
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        print('\lambda_0: {}'.format(args.lambda_0))
        print('\lambda_now: {}'.format(args.lambda_0 * (epoch / args.epochs)))
        print('epoch:', epoch)

        train(train_loader, model, criterion_isda, optimizer, epoch, args)

        print('=========> Save Covariance ')
        import pandas as pd
        var = np.array(criterion_isda.estimator.CoVariance.cpu(), dtype=np.float)
        for i in range(var.shape[0]):
            csv_name = os.path.join(args.train_url, 'Covariance/', '{0}_cov_imagenet.csv'.format(i))
            f = open(csv_name, 'w')
            for j in range(var.shape[1]):
                f.write(str(var[i][j]) + '\n')
            f.close()



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

    # model.train()
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        ages = images.cuda()
        target = target.cuda(non_blocking=True)

        output, features = model(images, isda=True)
        loss, output = criterion(model, images, target, args.lambda_0 * (epoch / args.epochs))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


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
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        print('lr:')
        print(param_group['lr'])


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