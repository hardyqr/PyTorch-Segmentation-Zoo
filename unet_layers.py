# Freddy @BH
# Feb 2nd, 2018

# reference: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_np 

def double_conv(in_ch,out_ch):
    conv_seq = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                )
    return conv_seq


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv,self).__init__()
        self.conv = double_conv(in_ch,out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class down_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_block,self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch,out_ch)
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_block,self).__init__()
        #self.up = nn.ConvTranspose2d(int(in_ch/2),int(in_ch/2),2,stride=2)
        self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv(in_ch,out_ch)
        
    def forward(self, x1, x2):
        '''
        x1: from last layer
        x2: from down conv process
        '''
        x = self.up(x1) # upconv, nxn->2nx2n
        #x2_cropped = F.pad(x2, )
        x2_cropped = x2
        x = torch.cat([x,x2_cropped],dim=1) # concat the channel dimension
        x = self.conv(x) # double conv
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv,self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,1) # 1x1 conv
    def forward(self, x):
        x = self.conv(x)
        return x
