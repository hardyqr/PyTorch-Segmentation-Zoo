
# modified from: https://raw.githubusercontent.com/clcarwin/focal_loss_pytorch/master/focalloss.py
# reference https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=1, size_average=True, ignore_index=0):
        super(FocalLoss2d, self).__init__()

        self.gamma = torch.tensor(gamma).cuda()
        self.weight = torch.tensor(weight).cuda()
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = self.weight
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -weight*((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
