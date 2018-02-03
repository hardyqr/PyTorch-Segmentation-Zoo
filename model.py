# Freddy @DP, uWaterloo
# Dec 4th, 2017

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd
#from tqdm import *
from data_preprocess import *
from utils import *

# Hyper params


# Handle data



# U-Net model
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=2),
            nn.ReLu()
            #nn.BatchNorm2d(16),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.ReLu()
            #nn.BatchNorm2d(32),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=2),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=2),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=2),
            nn.ReLU())
        self.fc = nn.Linear(8*8*16, 3)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        res_layer = self.layer5(out1)
        #print(res_layer.size())
        #print(out4.size())
        out5 = res_layer+out4
        out6 = out5.view(out5.size(0), -1)
        #print(out6.size())
        out7 = self.fc(out6)
        return out7
        




