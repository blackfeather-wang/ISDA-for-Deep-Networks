# -*-coding:utf-8-*-

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['shake_resnet26_2x32d', 'shake_resnet26_2x64d']


class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)

        return beta * grad_output, (1 - beta) * grad_output, None


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1,
                               stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1,
                               stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)

        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)

        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)

        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class ShakeBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(
            in_ch, out_ch, stride=stride)

        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):

    def __init__(self, depth, base_width, num_classes):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6

        in_chs = [16, base_width, base_width * 2, base_width * 4]
        self.in_chs = in_chs

        self.conv_1 = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(in_chs[0])

        self.stage_1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.stage_2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
        self.stage_3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)

        self.feature_num = in_chs[3]
        # self.fc_out = nn.Linear(in_chs[3], num_classes)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_chs[3])

        # out = self.fc_out(out)
        return out

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for _ in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


def shake_resnet26_2x32d(num_classes):
    return ShakeResNet(depth=26, base_width=32, num_classes=num_classes)


def shake_resnet26_2x64d(num_classes):
    return ShakeResNet(depth=26, base_width=64, num_classes=num_classes)

def shake_resnet26_2x112d(num_classes):
    return ShakeResNet(depth=26, base_width=112, num_classes=num_classes)



class ShakeBottleNeck(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        super(ShakeBottleNeck, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = None if self.equal_io else Shortcut(in_ch, out_ch, stride=stride)

        self.branch1 = self._make_branch(in_ch, mid_ch, out_ch, cardinary, stride)
        self.branch2 = self._make_branch(in_ch, mid_ch, out_ch, cardinary, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, padding=0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, stride=stride, groups=cardinary, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, out_ch, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNeXt(nn.Module):

    def __init__(self, depth, w_base, cardinary, label):
        super(ShakeResNeXt, self).__init__()
        n_units = (depth - 2) // 9
        n_chs = [64, 128, 256, 1024]
        self.n_chs = n_chs
        self.in_ch = n_chs[0]

        self.c_in = nn.Conv2d(3, n_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, n_chs[0], w_base, cardinary)
        self.layer2 = self._make_layer(n_units, n_chs[1], w_base, cardinary, 2)
        self.layer3 = self._make_layer(n_units, n_chs[2], w_base, cardinary, 2)
        self.feature_num = n_chs[3]
        # self.fc_out = nn.Linear(n_chs[3], label)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.n_chs[3])
        # h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, n_ch, w_base, cardinary, stride=1):
        layers = []
        mid_ch, out_ch = n_ch * (w_base // 64) * cardinary, n_ch * 4
        for i in range(n_units):
            layers.append(ShakeBottleNeck(self.in_ch, mid_ch, out_ch, cardinary, stride=stride))
            self.in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


def shake_resnext29_2x4x64d(num_classes):
    return ShakeResNeXt(depth=29, w_base=64, cardinary=4, label=num_classes)


