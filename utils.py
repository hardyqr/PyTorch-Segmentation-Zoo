# Freddy @Blair House
# Nov. 19, 2017
# edited, Feb. 3, 2018

# reference: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from __future__ import print_function, division
import os
import sys
from tqdm import *
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
#import image_slicer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class img_dataset():
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
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        #print(label.size())
        #print(label)
        return image, label

def show_imgs(image, labels):
    """Show image with landmarks"""
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(2)  # pause a bit so that plots are updated


def to_np(x):
    return x.data.cpu().numpy()

'''data to tensor, and save tensor'''

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

