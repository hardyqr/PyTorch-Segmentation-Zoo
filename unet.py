# Freddy @BH
# Feb 3rd, 2018

# reference: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from unet_layers import *

''' deeper u-net '''
class Unet(nn.Module):
    def __init__(self, channels_in, classes_out):
        super(Unet, self).__init__()
        self.inc = inconv(channels_in, 64)
        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 1024)
        self.down5 = down_block(1024, 2048)
        self.down6 = down_block(2048, 2048)
        self.up1 = up_block(4096, 1024)
        self.up2 = up_block(2048, 512)
        self.up3 = up_block(1024, 256)
        self.up4 = up_block(512, 128)
        self.up5 = up_block(256, 64)
        self.up6 = up_block(128, 64)
        self.outc = outconv(64, classes_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x= self.up1(x7, x6)
        x = self.up2(x, x5)
        #print("after up1, before up2: ",x.shape)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        #print(x.shape)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        #print(x.shape)
        x = self.outc(x)
        return x

''' shallow u-net '''
'''
class Unet(nn.Module):
    def __init__(self, channels_in, classes_out):
        super(Unet, self).__init__()
        self.inc = inconv(channels_in, 64)
        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 1024)
        self.down5 = down_block(1024, 1024)
        self.up1 = up_block(2048, 512)
        self.up2 = up_block(1024, 256)
        self.up3 = up_block(512, 128)
        self.up4 = up_block(256, 64)
        self.up5 = up_block(128, 64)
        self.outc = outconv(64, classes_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x= self.up1(x6, x5)
        x = self.up2(x, x4)
        #print("after up1, before up2: ",x.shape)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        #print(x.shape)
        x = self.up5(x, x1)
        #print(x.shape)
        x = self.outc(x)
        return x

'''


if __name__ == "__main__":
    """
    testing
    """
    model = Unet(3,1)
    x = Variable(torch.FloatTensor(np.random.random((1, 3, 512, 512))))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
