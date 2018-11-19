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
#from osgeo import gdal
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.metrics import confusion_matrix  
#import image_slicer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

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


def compute_pixel_acc(pred,gt,ignore_label=-1):
    """
    Args:
        pred - np.array
        gt - np.array
        """
    pixel_acc = None
    if ignore_label != -1:
        valid_mask = np.ones(gt.shape).astype(bool)
        valid_mask[gt==ignore_label] = False
        total = valid_mask.sum()
        correct_pixel = (pred*valid_mask==gt*valid_mask).sum()-len(gt)+total
        pixel_acc = correct_pixel / total
    else:
        pixel_acc = (pred==gt).sum()/len(gt)
    return pixel_acc


# reference: https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/c2b787483ed77b800336319a78e44df1d37e5407/pytorch_segmentation_detection/metrics.py
class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=255):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        
        if self.overall_confusion_matrix is not None:
            
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_current_mean_intersection_over_union(self):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union



def spy_sparse2torch_sparse(data):
    """
    https://discuss.pytorch.org/t/better-way-to-forward-sparse-matrix/21915/2
    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t
