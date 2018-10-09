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
import uuid
import sys
import math
import argparse
from distutils.version import LooseVersion
import argparse
from sklearn.metrics import jaccard_similarity_score

from utils import *
from unet import *
from duc_hdc import ResNetDUC, ResNetDUCHDC
from sunrgbd_data_loader import SUN_RGBD_dataset_train,SUN_RGBD_dataset_val, sunrgbd_drawer


parser = argparse.ArgumentParser()
parser.add_argument('--train_rgb_path', default='~/data/standord-indoor/area_1/data/rgb/')
parser.add_argument('--train_depth_path', default='~/data/standord-indoor/area_1/data/rgb/')
parser.add_argument('--train_label_path', default='~/data/standord-indoor/area_1/data/semantic/')
parser.add_argument('--val_rgb_path', default='~/data/standord-indoor/area_2/data/rgb/')
parser.add_argument('--val_depth_path', default='~/data/standord-indoor/area_2/data/rgb/')
parser.add_argument('--val_label_path', default='~/data/standord-indoor/area_2/data/semantic/')
parser.add_argument('--batch_size', default='4',type=int)
parser.add_argument('--image_size', default='512',type=int)
parser.add_argument('--n_classes', default='13',type=int)
parser.add_argument('--load_prev', default='none',type=str)
parser.add_argument('--lr', default='0.001',type=float)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--use_depth', action='store_true')
parser.add_argument('--resume', default='',type=str)

opt = parser.parse_args()
print(opt)

# Hyper params
num_epochs = 30
batch_size = opt.batch_size
learning_rate = opt.lr

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


train_set = SUN_RGBD_dataset_train(opt.train_rgb_path, opt.train_depth_path, opt.train_label_path)
        #transform=transforms.Compose([
           # transforms.RandomHorizontalFlip(), 
            #transforms.RandomResizedCrop(opt.image_size), 
            #transforms.RandomCrop((640,480)),
            #transforms.Resize((512,512)),
            #transforms.ToTensor()]))

val_set = SUN_RGBD_dataset_val(opt.val_rgb_path, opt.val_depth_path, opt.val_label_path,
        transform=transforms.Compose([
            transforms.Resize((640,480)),
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

#unet = Unet(4,13)
n_classes = opt.n_classes
model = ResNetDUCHDC(n_classes)

if(opt.load_prev != 'none'):
    model.load_state_dict(torch.load(opt.load_prev))

if(use_gpu):
    model.cuda()

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

counter = 0

def validate(iter_num=None, early_break=False):
    ''' val '''
    #print('Validation: ')
    accs = []
    IoUs = []
    for i,(image, depth, label,name) in enumerate(val_loader):
        # stack rgb and depth
        stacked = None
        if opt.use_depth:
            stacked = torch.cat((image,depth/120),1)
            stacked = Variable(stacked)
        else:
            stacked = Variable(image)
        labels = Variable(label).type('torch.LongTensor').squeeze(1)

        if use_gpu:
            stacked = stacked.cuda()
            labels = label.cuda()
        outputs = model(stacked)

        
        # acc 
        #acc = simple_acc(outputs,labels)# Truth_Positive / NumberOfPixels
        acc = compute_pixel_acc(outputs,labels)
        #iou = compute_iou(outputs,labels)
        #iou = jaccard_similarity_score(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        #print ('i acc:{:.3f}'.format(100*acc))
        accs.append(acc)
        IoUs.append(0)
        if i % 100 == 0 and i != 0:
            print('current sample: {}, pixelAcc so far: {:.3f}% mIoU so far: {:.3f}'.format 
                    (i, 100*sum(accs)/float(len(accs)),100*sum(IoUs)/float(len(IoUs))))
            # take one sample out for visualization
            d = sunrgbd_drawer()
            k = np.random.randint(int(outputs.shape[0]))
            _,_outputs = outputs.max(1)
            #print (_outputs.size(), labels.size())
            output = to_np(_outputs[k])
            label = to_np(labels.squeeze(1)[k])
            image_name = name[k].split('.')[0]
            output_rgb = d.decode_segmap(output,n_classes)
            label_rgb = d.decode_segmap(label,n_classes)
            #print (output_rgb.shape,label_rgb.shape)
            im1 = Image.fromarray(output_rgb.astype('uint8'))
            key = uuid.uuid4().hex.upper()[0:6]
            if iter_num is None:iter_num = ""
            im1.save(os.path.join('./generated_masks/','pred_'+image_name+'_'+str(iter_num)+'_'+key+'.png'))
            im2 = Image.fromarray(label_rgb.astype('uint8'))
            im2.save(os.path.join('./generated_masks/','gt_'+image_name+'_'+str(iter_num)+'_'+key+'.png'))

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
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/10)
    elif (epoch == 15):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/20)
    elif (epoch == 25):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/100)
    #"""
    #validate(True)
    for i,(image,depth,label) in enumerate(train_loader):
        if(debug and counter>=3):break
        counter+=1
        #print(counter)
        stacked = None
        if opt.use_depth:
            stacked = torch.cat((image,depth/120),1)
            stacked = Variable(stacked)
        else:
            stacked = Variable(image)

        labels = Variable(label).type('torch.LongTensor').squeeze(1)

        if(use_gpu):
            images = stacked.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        #target = labels.view(-1, )
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        #loss = dice_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        #validate(True)

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Batch [%d/%d] Loss: %.6f'
                   %(epoch+1, num_epochs,i+1, len(train_loader),loss.data[0]))
            validate(i,True)
        if (i+1) % 5000 == 0:
            validate(i,False)
            # save Model
            if (not debug):
                torch.save(model.state_dict(),os.path.join('~/models/','epoch_'+str(i)+'_model.pkl'))


       
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





