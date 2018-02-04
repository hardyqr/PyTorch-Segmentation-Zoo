# Freddy @DP&BH, uWaterloo
# Feb 4, 2018

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import *
import sys
import math
import argparse
from distutils.version import LooseVersion

from utils import *
from unet import *


# Hyper params
num_epochs = 1000
batch_size = 4

# global vars
#parser = argparse.ArgumentParser(description='Short sample app')
#parser.add_argument('-d', action="debug_mode", dest='debug',default=False)
debug = False
#debug = True
if(torch.cuda.is_available()):
    use_gpu = True


# Handle data
train_set =img_dataset(sys.argv[1],sys.argv[2],
        transform=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()]))

# Handle data
test_set =img_dataset(sys.argv[1],sys.argv[2],
        transform=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=4, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=True)
# 


unet = Unet(3,3)

if(use_gpu):
    unet.cuda()

criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)



counter = 0

print("len(train_loader) :" ,len(train_loader))
# Train the Model
for epoch in range(num_epochs):
    if(debug and counter>=3):
        break
    for i,(image,label) in enumerate(train_loader):
        if(debug and counter>=3):break
        counter+=1
        #print(counter)
        images = Variable(image)
        labels = Variable(label.float())

        test = to_np(labels)[0].transpose((1,2,0))
        #print(test.shape)
        #print(test*255)
        #test = Image.fromarray(np.uint8(test*255))
        #test.show()
        #print(to_np(labels))

        if(use_gpu):
            images = images.cuda()
            labels = labels.cuda()
        #print(images.size(), labels.size())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = unet(images)
        #print(outputs.size(), labels.size())
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        if (i+1) % 1 == 0:
            print ('Epoch [%d/%d], Batch [%d/%d] Loss: %.4f'
                   %(epoch+1, num_epochs,i+1, len(train_loader),loss.data[0]))


''' test '''

for i, (image, label) in enumerate(test_loader):
    image = Variable(image).cuda()
    img = unet(image)
    img = np.uint8(to_np(img)[0].transpose((1,2,0))*255)
    print(img.shape)
    img = Image.fromarray(img, 'RGB')
    #img.show()
    img.save('./outputs/test' + str(i)+ '.png')
