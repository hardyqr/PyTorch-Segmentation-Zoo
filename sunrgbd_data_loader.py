
from __future__ import print_function, division
import os
import sys
from tqdm import *
import pandas as pd
import numpy as np
from numpy import random
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from skimage import io, transform
#import image_slicer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

class SUN_RGBD_dataset_train():
    """ SUN-RGBD dataset - RGB-D semantic labeling.
    Download datset from : http://cvgl.stanford.edu/data2/sun_rgbd.tgz
    """

    def __init__(self,img_dir,depth_dir,mask_dir,transform=None):
        """
        Args:
            mask_dir (string): Path to (img) annotations.
            img_dir (string): Path with all the training images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)
        self.depth_names = os.listdir(depth_dir)
        self.label_names = os.listdir(mask_dir)
        
        self.img_names.sort()
        self.depth_names.sort()
        self.label_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%idx)
        image = Image.open(os.path.join(self.img_dir,self.img_names[idx])).convert('RGB')
        depth = Image.open(os.path.join(self.depth_dir,self.depth_names[idx]))
        label = Image.open(os.path.join(self.mask_dir,self.label_names[idx]))

        if self.transform:
            image = self.transform(image).type('torch.FloatTensor')
            depth = self.transform(depth).type('torch.FloatTensor')
            label = self.transform(label).type('torch.FloatTensor')
        return image, depth, label


class SUN_RGBD_dataset_val():
    """SUN-RGBD image semantic labeling dataset."""

    def __init__(self,img_dir,depth_dir,mask_dir,transform=None):
        """
        Args:
            mask_dir (string): Path to (img) annotations.
            img_dir (string): Path with all the training images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)
        self.depth_names = os.listdir(depth_dir)
        self.label_names = os.listdir(mask_dir)

        self.img_names.sort()
        self.depth_names.sort()
        self.label_names.sort()


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%idx)
        image = Image.open(os.path.join(self.img_dir,self.img_names[idx])).convert('RGB')
        depth = Image.open(os.path.join(self.depth_dir,self.depth_names[idx]))
        label = Image.open(os.path.join(self.mask_dir,self.label_names[idx]))
       
        if self.transform:
            image = self.transform(image).type('torch.FloatTensor')
            depth = self.transform(depth).type('torch.FloatTensor')
            label = self.transform(label).type('torch.FloatTensor')
        #print(label.size())
        #return image, label, name, (original_image.size[0],original_image.size[1])
        return image, depth, label


def show_imgs(image, labels):
    """Show image with landmarks"""
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(2)  # pause a bit so that plots are updated


def random_transpose(image, label):
    methods = [PIL.Image.FLIP_LEFT_RIGHT,
                    PIL.Image.FLIP_TOP_BOTTOM,
                    PIL.Image.ROTATE_90,
                    PIL.Image.ROTATE_180,
                    PIL.Image.ROTATE_270,
                    PIL.Image.TRANSPOSE]
    r = random.randint(0,len(methods)-1)
    method = methods[r]
    if(random.randint(0,2) != 0):# 1/3 keep
        image = image.transpose(method) # transpose
        label = label.transpose(method) # transpose
    return image, label

def random_crop(PIL_img,label,ratio):
    """
    Args:
        PIL_img: image in PIL format.
        ratio: 0 < ratio <= 1.
    output:
        A PIL formt image with side length ratio*original side length.
    """
    (width, height) = PIL_img.size
    h_shift = np.random.randint(-height*(1-ratio)/2+1,height*(1-ratio)/2-1)
    w_shift = np.random.randint(-width*(1-ratio)/2+1,width*(1-ratio)/2-1)
    new_center = (int(height/2)+h_shift,int(width/2)+w_shift)
    cropped_area = (
                new_center[1] - ratio*width/2,
                new_center[0] - ratio*height/2,
                new_center[1] + ratio*width/2,
                new_center[0] + ratio*height/2
                )
    return PIL_img.crop(cropped_area), label.crop(cropped_area)

def random_rotate(PIL_img, label, _range):
    (width, height) = PIL_img.size
    angle = np.random.randint(-_range,_range)
    ratio = 0.7
    img = PIL_img.rotate(angle)
    label = label.rotate(angle)
    center = (int(height/2),int(width/2))
    cropped_area = (
                center[1] - ratio*width/2,
                center[0] - ratio*height/2,
                center[1] + ratio*width/2,
                center[0] + ratio*height/2
                )   
    return img.crop(cropped_area), label.crop(cropped_area)



def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# Dice Loss - a loss for multi-class segmentation task
# https://github.com/pytorch/pytorch/issues/1249
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

