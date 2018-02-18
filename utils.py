# Freddy @Blair House
# Nov. 19, 2017
# edited, Feb. 3, 2018


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

# reference: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class img_dataset_train():
    """vaihingen image semantic labeling dataset."""

    def __init__(self, mask_dir, img_dir, transform=None):
        """
        Args:
            mask_dir (string): Path to (img) annotations.
            img_dir (string): Path with all the training images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform
        names = os.listdir(img_dir)
        names.sort()
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%idx)
        image = Image.open(self.img_dir+'/'+self.names[idx])
        label = Image.open(self.mask_dir+'/'+self.names[idx])
        # data augmentation
        image,label = augmentation_tool(image, label)
        #image.save(str(idx)+'1.png')
        #label.save(str(idx)+'2.png')
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        # avoid inaccurate rgb labels
        label[ label[:,:,:] >= 0.5] = 1  
        label[ label[:,:,:] < 0.5] = 0
        #Image.fromarray((onehot2rgb(rgb2onehot2(label))*255)).save('./test.png')
        #print(label.size())
        return image, label

class img_dataset_test():
    """vaihingen image semantic labeling dataset."""

    def __init__(self, mask_dir, img_dir, transform=None):
        """
        Args:
            mask_dir (string): Path to (img) annotations.
            img_dir (string): Path with all the training images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform
        names = os.listdir(img_dir)
        names.sort()
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%idx)
        original_image = Image.open(self.img_dir+'/'+self.names[idx])
        label = Image.open(self.mask_dir+'/'+self.names[idx])

        if self.transform:
            image = self.transform(original_image)
            label = self.transform(label)
        label[ label[:,:,:] >= 0.5] = 1
        label[ label[:,:,:] < 0.5] = 0
        #print(label.size())
        return image, label, (original_image.size[0],original_image.size[1])


def show_imgs(image, labels):
    """Show image with landmarks"""
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(2)  # pause a bit so that plots are updated


def augmentation_tool(image, label):
    # data augmentation - random crop
    if(random.randint(0,2) == 0):# 2/3 keep
        image,label = random_crop(image,label,0.8)
    # data augmentation - affine transformation
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

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def rgb2onehot(labels):
    """
    Args:
        labels: A torch Variable.
    Outputs:
        A torch Variavle.
    """
    labels = to_np(labels).transpose((0,2,3,1))
    s = labels.shape
    onehot_labels = np.zeros((s[0],s[1],s[2],6))
    l1,l2,l3,l4,l5,l6 = np.zeros(s),np.zeros(s),np.zeros(s),np.zeros(s),np.zeros(s),np.zeros(s)
    l1 += np.array([1,1,1]) # impervious surface
    l2 += np.array([0,0,1]) # building
    l3 += np.array([0,1,1]) # low vegetation
    l4 += np.array([0,1,0]) # tree
    l5 += np.array([1,1,0]) # car
    l6 += np.array([1,0,0]) # clutter
    mask1 = np.sum((labels == l1).astype(np.float32), axis=-1)==3
    mask2 = np.sum((labels == l2).astype(np.float32), axis=-1)==3
    mask3 = np.sum((labels == l3).astype(np.float32), axis=-1)==3
    mask4 = np.sum((labels == l4).astype(np.float32), axis=-1)==3
    mask5 = np.sum((labels == l5).astype(np.float32), axis=-1)==3
    mask6 = np.sum((labels == l6).astype(np.float32), axis=-1)==3
    ll1,ll2,ll3,ll4,ll5,ll6 = np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6)
    ll1[0], ll2[1], ll3[2], ll4[3], ll5[4], ll6[5] = 1., 1., 1., 1., 1., 1.
    onehot_labels[mask1] = ll1
    onehot_labels[mask2] = ll2
    onehot_labels[mask3] = ll3
    onehot_labels[mask4] = ll4
    onehot_labels[mask5] = ll5
    onehot_labels[mask6] = ll6
    return Variable(torch.from_numpy(onehot_labels.transpose((0,3,1,2))).float())

def onehot2rgb(predict):
    """
    Args:
        predict: A torch Variable.
    Outputs:
        A numpy array.
    """
    predict =  to_np(predict).transpose([0,2,3,1])
    s = predict.shape
    label = np.argmax(predict, axis=-1)
    rgb =  np.zeros((s[0],s[1],s[2],3))
    rgb[ label[:,:,:] == 0. ] = np.array([1,1,1])
    rgb[ label[:,:,:] == 1. ] = np.array([0,0,1])
    rgb[ label[:,:,:] == 2. ] = np.array([0,1,1])
    rgb[ label[:,:,:] == 3. ] = np.array([0,1,0])
    rgb[ label[:,:,:] == 4. ] = np.array([1,1,0])
    rgb[ label[:,:,:] == 5. ] = np.array([1,0,0])
    return rgb


'''
if __name__ == "__main__":
    transformed_dataset = img_dataset(
        sys.argv[1],
        sys.argv[2],
        transform=transforms.Compose([ToTensor()]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['label'].size())

        if i == 3:
            break

'''
if __name__ == "__main__":
    i = np.array(PIL.Image.open('gts.png'))
    i = i[:,:,0:3]
    #PIL.Image.fromarray(i,'RGB').show()
    i = i.transpose((2,0,1))
    ii = np.array([ np.array(i)/255 ])
    i = to_var(torch.from_numpy(ii)) # PngImageFile to torch Variable
    t = rgb2onehot(i)

    tt = np.array(onehot2rgb(t)[0])
    p = PIL.Image.fromarray((tt*255).astype('uint8'))
    # p.show()
    p.save('sample.png')

