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
import argparse

from utils import *
from unet import *
from sunrgbd_data_loader import SUN_RGBD_dataset_train,SUN_RGBD_dataset_val


parser = argparse.ArgumentParser()
parser.add_argument('--train_rgb_path', default='~/data/standord-indoor/area_1/data/rgb/')
parser.add_argument('--train_depth_path', default='~/data/standord-indoor/area_1/data/rgb/')
parser.add_argument('--train_label_path', default='~/data/standord-indoor/area_1/data/semantic/')
parser.add_argument('--val_rgb_path', default='~/data/standord-indoor/area_2/data/rgb/')
parser.add_argument('--val_depth_path', default='~/data/standord-indoor/area_2/data/rgb/')
parser.add_argument('--val_label_path', default='~/data/standord-indoor/area_2/data/semantic/')
parser.add_argument('--batch_size', default='4',type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--resume', default='',type=str)

opt = parser.parse_args()
print(opt)

# Hyper params
num_epochs = 30
batch_size = opt.batch_size
learning_rate = 1e-4

# global vars
#parser = argparse.ArgumentParser(description='Short sample app')
#parser.add_argument('-d', action="debug_mode", dest='debug',default=False)
debug = False
#debug = True
load_prev = True
load_prev = False
if(torch.cuda.is_available()):
    use_gpu = True


# Handle data

train_set = SUN_RGBD_dataset_train(opt.train_rgb_path, opt.train_depth_path, opt.train_label_path,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomResizedCrop(512), 
            #transforms.Resize((512,512)),
            transforms.ToTensor() ]))

val_set = SUN_RGBD_dataset_val(opt.val_rgb_path, opt.val_depth_path, opt.val_label_path,
        transform=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()]))

#test_set =img_dataset_test(sys.argv[5],
#        transform=transforms.Compose([
#            transforms.Resize((512,512)),
#            transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size, shuffle=False)

#test_loader = torch.utils.data.DataLoader(
#        test_set,
#        batch_size=1, shuffle=False)

unet = Unet(4,13)

if(load_prev):
    unet.load_state_dict(torch.load('prev_model.pkl'))

if(use_gpu):
    unet.cuda()

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

counter = 0


def compute_pixel_acc(pred,gt):
    """
    Args:
        pred - tensor of size (batch_size, n_classes, H, W)
        gt - tensor of size (batch_size, H, W)
    """
    _,pred = pred.max(1)
    _gt = to_np(gt)
    _pred = to_np(pred)
    return (_pred==_gt).sum()/len(gt.view(-1))

def validate(early_break=False):
    ''' val '''
    #print('Validation: ')
    accs = []
    for i,(image, depth, label) in enumerate(val_loader):
        # stack rgb and depth
        stacked = torch.cat((image,depth/120),1)
        stacked = Variable(stacked)
        labels = Variable(label).type('torch.LongTensor').squeeze(1)

        if use_gpu:
            stacked = stacked.cuda()
            labels = label.cuda()
        outputs = unet(stacked)
        
        # acc 
        #acc = simple_acc(outputs,labels)# Truth_Positive / NumberOfPixels
        acc = compute_pixel_acc(outputs,labels)
        accs.append(acc)
        if i % 100 == 0 and i != 0:
            print('current sample: {}, pixelAcc so far: {:.3f}%'.format (i, 100*sum(accs)/float(len(accs))))
            if early_break:
                return None

        #img = np.uint8(img[0]*255)
        #img = Image.fromarray(img, 'RGB')
        #img = img.resize((w.numpy(),h.numpy()))
        #img.show()
        #img.save(sys.argv[6] + '/val_' + str(epoch) + '_' + str(i)+ '.png')
    print('Overall pixelAcc - all pics: %.3f%%' % (100*sum(accs)/float(len(accs))))

print("len(train_loader) :" ,len(train_loader))
# Train the Model
for epoch in range(num_epochs):
    #if(debug and counter>=3): break
    #"""
    if (epoch == 5):
        optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate/10)
    elif (epoch == 15):
        optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate/20)
    elif (epoch == 25):
        optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate/100)
    #"""
    #validate()
    for i,(image,depth,label) in enumerate(train_loader):
        if(debug and counter>=3):break
        counter+=1
        #print(counter)
        stacked = torch.cat((image,depth/120),1)
        stacked = Variable(stacked)
        labels = Variable(label).type('torch.LongTensor').squeeze(1)

        if(use_gpu):
            images = stacked.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = unet(images)
        #target = labels.view(-1, )
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        #loss = dice_loss(outputs, labels)
        loss.backward()
        optimizer.step()


        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Batch [%d/%d] Loss: %.6f'
                   %(epoch+1, num_epochs,i+1, len(train_loader),loss.data[0]))
        if (i+1) % 1000 == 0:
            validate(True)


       
    ''' test '''
    """
    print('running test set... ')
    for i,(image,(w,h)) in enumerate(tqdm(test_loader)):
        image = Variable(image).cuda()
        img = unet(image)
        #acc = simple_acc(outputs,labels)# Truth_Positive / NumberOfPixels
        #print(acc)
        img = onehot2rgb(img)
        img = np.uint8(img[0]*255)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((w.numpy(),h.numpy()))
        #img.show()

        img.save(sys.argv[6] + '/test_' + str(epoch) + '_' + str(i)+ '.png')
    """

    # save Model
    if (not debug):
        torch.save(unet.state_dict(),'prev_model.pkl')



