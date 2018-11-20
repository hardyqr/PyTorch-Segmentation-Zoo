
# reference: https://raw.githubusercontent.com/BardOfCodes/pytorch_deeplab_large_fov/master/deeplab_large_fov.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class deeplab_largeFOV(nn.Module):
    def __init__(self, n_labels=21):
        super(deeplab_largeFOV, self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1,bias=False)
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1,bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2,padding=1)
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1,bias=False)
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1,bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1,bias=False)
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1,bias=False)
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1,bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1,bias=False)
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1,bias=False)
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1,bias=False)
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 2,dilation = 2,bias=False)
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 2,dilation = 2,bias=False)
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 2,dilation = 2,bias=False)
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size = 3, stride = 1,padding=1)
        self.fc6 = nn.Conv2d(512,1024,3,padding = 12,dilation = 12,bias=False)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024,1024,1,bias=False)
        self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8 = nn.Conv2d(1024,n_labels,1,bias=False)
        self.log_softmax = nn.LogSoftmax()
        self.fc8_interp_test = nn.UpsamplingBilinear2d(size=(513,513))
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(F.relu(self.conv4_3(x)))
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5a(self.pool5(F.relu(self.conv5_3(x))))
        x = self.drop6(F.relu(self.fc6(x)))
        x = self.drop7(F.relu(self.fc7(x)))
        x = self.fc8(x)
        
        return x
    
    def forward_test(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(F.relu(self.conv4_3(x)))
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5a(self.pool5(F.relu(self.conv5_3(x))))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8_voc12(x)
        x = self.log_softmax(x)
        x = self.fc8_interp_test(x)
        
        return x
