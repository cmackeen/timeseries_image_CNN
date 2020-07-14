from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.functional as W



def basic(x, size, stride):
    y = W.conv(x, size, 3, stride=stride, padding=1, bias=False)
    y = W.batch_norm(y, activation='relu')
    y = W.conv(y, size, 3, stride=1, padding=1, bias=False)
    y = W.batch_norm(y)
    if y.shape[1] != x.shape[1]: # channel size mismatch, needs projection
        x = W.conv(x, y.shape[1], 1, stride=stride, bias=False)
        x = W.batch_norm(x)
    y = y+x # residual shortcut connection
    return F.relu(y)


def stack(x, num_block, size, stride, block=basic):
    for s in [stride]+[1]*(num_block-1):
        x = block(x, size, s)
    return x


class ResNet(nn.Module):

    def __init__(self, block=basic,
            stack_spec=((2, 64, 1), (2, 128, 2), (2, 256, 2), (2, 512, 2))):
        super().__init__()
        self.block = block
        self.stack_spec = stack_spec
        warm.up(self, [2, 3, 32, 32])

    def forward(self, x):
        y = W.conv(x, 64, 7, stride=2, padding=3, bias=False)
        y = W.batch_norm(y, activation='relu')
        y = F.max_pool2d(y, 3, stride=2, padding=1)
        for spec in self.stack_spec:
            y = stack(y, *spec, block=self.block)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = W.linear(y, 2000)
        return y
    
