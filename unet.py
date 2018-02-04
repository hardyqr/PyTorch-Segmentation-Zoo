# Freddy @BH
# Feb 3rd, 2018

# reference: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from unet_layers import *


class Unet(nn.Module):
    def __init__(self, channels_in, classes_out):
        super(Unet, self).__init__()
        self.inc = inconv(channels_in, 64)
        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 512)
        self.up1 = up_block(1024, 256)
        self.up2 = up_block(512, 128)
        self.up3 = up_block(256, 64)
        self.up4 = up_block(128, 64)
        self.outc = outconv(64, classes_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #print(x4.shape)
        x5 = self.down4(x4)
        #print("before up1: ", x5.shape)
        x = self.up1(x5, x4)
        #print("after up1, before up2: ",x.shape)
        x = self.up2(x, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #print(x.shape)
        x = self.outc(x)
        #print(x.shape)
        return x


if __name__ == "__main__":
    """
    testing
    """
    model = Unet(3,1)
    x = Variable(torch.FloatTensor(np.random.random((1, 3, 500, 500))))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
