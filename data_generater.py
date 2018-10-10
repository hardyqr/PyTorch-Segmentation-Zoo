# Feb 19, 2018 @BH

# data augmentation for isprs 2d segmentation (Vaihingen) benchmark dataset

from utils import *
import sys
from PIL import Image
import PIL
import numpy as np
from tqdm import tqdm

image_in_folder = sys.argv[1]
label_in_folder = sys.argv[2]
names = os.listdir(image_in_folder)
image_out_folder = sys.argv[3]
label_out_folder = sys.argv[4]


if __name__ == '__main__':
    count = 4701
    for i in tqdm(range(0,200)):
        for f in names:
            #print(f)
            count += 1
            mymy = str(count)
            i = Image.open(image_in_folder + '/' + f)
            l = Image.open(label_in_folder + '/' + f)
            '''affine transformation'''

            # rotate and crop
            if (np.random.randint(0,3) == 0):
                i, l = random_rotate(i, l, 15)
            elif (np.random.randint(0,2) == 0):
                ratio = np.random.randint(80,100)/100
                i, l = random_crop(i, l, ratio)

            # transpose
            if (np.random.randint(0,2) == 0):
                i, l = random_transpose(i, l)


            i.save(image_out_folder+'/testing_'+f+'_'+ mymy +'.png', 'PNG')
            l.save(label_out_folder+'/testing_'+f+'_'+ mymy +'.png', 'PNG')
