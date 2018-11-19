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
from tensorboardX import SummaryWriter


from utils import *
from models.unet import *
from models.duc_hdc import ResNetDUC, ResNetDUCHDC
from loss.focalloss2d import FocalLoss2d 
from KernelCut.kernelcut import *
#from models.deeplab_resnet import Res_Deeplab
from models.deeplab_resnet_2 import Res_Deeplab
from dataloaders.sunrgbd_data_loader import SUN_RGBD_dataset_train,SUN_RGBD_dataset_val, sunrgbd_drawer
from dataloaders.pascal_voc_data_loader import pascal_voc_dataset_train, pascal_voc_dataset_val


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='~/data/standord-indoor/')
parser.add_argument('--log_path', default='None')
parser.add_argument('--train_rgb_path', default='~/data/standord-indoor/area_1/data/rgb/')
parser.add_argument('--train_depth_path', default='~/data/standord-indoor/area_1/data/rgb/')
parser.add_argument('--train_label_path', default='~/data/standord-indoor/area_1/data/semantic/')
parser.add_argument('--train_seeds_path', default='~/data/standord-indoor/area_1/data/seeds/')
parser.add_argument('--val_rgb_path', default='~/data/standord-indoor/area_2/data/rgb/')
parser.add_argument('--val_depth_path', default='~/data/standord-indoor/area_2/data/rgb/')
parser.add_argument('--val_label_path', default='~/data/standord-indoor/area_2/data/semantic/')
parser.add_argument('--pascal_train_file', default='./train.txt')
parser.add_argument('--pascal_val_file', default='~/val.txt')
parser.add_argument('--model', default='ResNetDUCHDC',help="select from {ResNetDUCHDC,UNet,Deeplab-v2}")
parser.add_argument('--data', default='SUNRGBD',help="select from {SUNRGBD,Pascal_VOC}")
parser.add_argument('--batch_size', default='4',type=int)
parser.add_argument('--image_size', default='512',type=int)
parser.add_argument('--n_classes', default='13',type=int)
parser.add_argument('--epochs', default='30',type=int)
parser.add_argument('--load_prev', default='none',type=str)
parser.add_argument('--lr', default='0.001',type=float)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--use_depth', action='store_true')
parser.add_argument('--weak_sup', action='store_true')
parser.add_argument('--resume', default='',type=str)

opt = parser.parse_args()
print(opt)

writer = None
if opt.log_path == 'None':
    writer = SummaryWriter()
else:
    writer = SummaryWriter(opt.log_path)


IGNORE_LABEL = 0

# Hyper params
num_epochs = opt.epochs
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
train_sizes = None #[(reisze_size),(crop_size)]
val_transform = None
train_set = None
val_set = None
sizes = None

# transforms
if opt.model == 'UNet':
    sizes = [(600,600),(512,512)]
elif opt.model == "Deeplab-v2":
    sizes = [(350,350),(321,321)]
else:
    sizes = [(480,560),(420,480)]

val_transform = transforms.Compose([
            transforms.Resize(sizes[1]),
            transforms.ToTensor()])

interp = nn.Upsample(size=sizes[1], mode='bilinear')

# choose dataloader
if opt.data == "SUNRGBD":
    train_set = SUN_RGBD_dataset_train(opt.train_rgb_path, opt.train_depth_path, opt.train_label_path,sizes=sizes)
    val_set = SUN_RGBD_dataset_val(opt.val_rgb_path, 
        opt.val_depth_path, opt.val_label_path,
        transform=val_transform,sizes=sizes)
elif opt.data == "Pascal_VOC":
    train_set = pascal_voc_dataset_train(opt.pascal_train_file,opt.data_root,sizes=sizes)
    val_set = pascal_voc_dataset_val(opt.pascal_val_file, opt.data_root,
            transform=val_transform,sizes=sizes)

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
model = None
if opt.model == "ResNetDUCHDC":
    model = ResNetDUCHDC(n_classes)
elif opt.model == "UNet":
    if opt.use_depth:
        model = Unet(4, n_classes)
    else:
        model = Unet(3,n_classes)
elif opt.model == "Deeplab-v2":
    #model = Res_Deeplab(num_classes=n_classes)
    model = Res_Deeplab(NoLabels=n_classes)
    # load pre-trained weights
    saved_state_dict = torch.load('/home/fangyu/data/pretrained_models/MS_DeepLab_resnet_trained_VOC.pth')
    #saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
    if n_classes!=21:
        for i in saved_state_dict:
            #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            #print (i_parts)
            #print (model.state_dict([i]))
            if i_parts[1]=='layer5':
                saved_state_dict[i] = model.state_dict()[i]
    model.load_state_dict(saved_state_dict)
    print ('[pretrained weights loaded!]')



if(opt.load_prev != 'none'):
    model.load_state_dict(torch.load(opt.load_prev))

if(use_gpu):
    model.cuda()

#criterion = nn.MSELoss()
criterion = FocalLoss2d(ignore_index=IGNORE_LABEL)
#criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9,weight_decay = 0.0005)

counter = 0

def validate(iter_num=None, early_break=False):
    ''' val '''
    #print('Validation: ')
    accs = []
    valid_label_list = None
    running_confusion_matrix = None
    if opt.data == "SUNRGBD" or opt.data == "Pascal_VOC":
        valid_label_list = [i for i in range(1,opt.n_classes)]
    if not early_break:
        running_confusion_matrix = RunningConfusionMatrix(labels=valid_label_list)
    for i,(image, depth, label, img_path) in enumerate(val_loader):
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
        if "Deeplab" in opt.model:
            outputs = interp(outputs) 
        #print (outputs)
        #sys.exit(0)

        '''compute metrics'''
        # acc 
        #acc = simple_acc(outputs,labels)# Truth_Positive / NumberOfPixels

        _, prediction = outputs.max(1)
        prediction = prediction.squeeze(1)
        prediction_np = to_np(prediction).flatten()
        annotation_np = to_np(labels).flatten()
        acc = compute_pixel_acc(prediction_np,annotation_np,ignore_label=IGNORE_LABEL)
        if not early_break: # only compute mIOU at the end of every epoch
            running_confusion_matrix.update_matrix(annotation_np, prediction_np)


        #miou = compute_iou(outputs,labels,labels=valid_label_list)
        #iou = jaccard_similarity_score(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        #print ('i acc:{:.3f}'.format(100*acc))
        accs.append(float(acc))

        # sample some image for visualization
        if i % random.randint(50,150) == 0 and i != 0:
            print('current sample: {}, pixelAcc so far: {:.3f}% '.format 
                    (i, 100*sum(accs)/float(len(accs))))
            # take one sample out for visualization
            d = sunrgbd_drawer()
            k = np.random.randint(int(outputs.shape[0]))
            _,_outputs = outputs.max(1)
            #print (_outputs.size(), labels.size())
            output = to_np(_outputs[k])
            label = to_np(labels.squeeze(1)[k])
            _img_path = img_path[k]
            image_name = img_path[k].split('/')[-1].split('.')[0]
            output_rgb = d.decode_segmap(output,n_classes)
            label_rgb = d.decode_segmap(label,n_classes)
            #print (output_rgb.shape,label_rgb.shape)
            im1 = Image.fromarray(output_rgb.astype('uint8'))
            key = uuid.uuid4().hex.upper()[0:6]
            if iter_num is None:iter_num = ""
            im1.save(os.path.join('./generated_masks/','pred_'+image_name+'_'+str(iter_num)+'_'+key+'.png'))
            im2 = Image.fromarray(label_rgb.astype('uint8'))
            im2.save(os.path.join('./generated_masks/','gt_'+image_name+'_'+str(iter_num)+'_'+key+'.png'))
            im3 = Image.open(_img_path).resize(sizes[1])
            im3.save(os.path.join('./generated_masks/','rgb_'+image_name+'_'+str(iter_num)+'_'+key+'.png'))

            del output
            del label
            del outputs
            del labels

            if early_break:
                if opt.debug:
                    print ('break loop')
                    break
                else:
                    return None

        #img = np.uint8(img[0]*255)
        #img = Image.fromarray(img, 'RGB')
        #img = img.resize((w.numpy(),h.numpy()))
        #img.show()
        #img.save(sys.argv[6] + '/val_' + str(epoch) + '_' + str(i)+ '.png')
    # TODO
    # send some generated masks to visdom
    #im3 = Image.open(_img_path)
    #writer.add_image('rgb_'+image_name,np.array(im3),global_step=iter_num)
    #writer.add_image('gt_'+image_name,np.array(im2),global_step=iter_num)
    #writer.add_image('pred_'+image_name,np/array(im1),global_step=iter_num)

    pixel_acc = sum(accs)/float(len(accs))
    print('Overall pixelAcc - all pics: %.3f%%' % (100*pixel_acc))
    if not early_break:
        mIoU = running_confusion_matrix.compute_current_mean_intersection_over_union()
        print('Overall mIOU - all pics: %.3f%%' % (100*mIoU))
    return pixel_acc, mIoU

print("len(train_loader) :" ,len(train_loader))
# Train the Model
for epoch in range(num_epochs):
    #if(debug and counter>=3): break
    """
    # learning rate decay by 10 times every 10 epochs
    if (epoch+1) % 10 == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/10)
    """
    if opt.debug:
        validate(-1,False)
    #for i,(image,depth,label,seeds) in enumerate(train_loader):
    for i,(image,depth,label,image_ori) in enumerate(train_loader):
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
        #print (outputs.size())
        if "Deeplab" in opt.model:
            outputs = interp(outputs) 
        #print (outputs.size())
        #target = labels.view(-1, )
        #sys.exit(0)
        #print(labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #validate(True)
        if (i+1) % 100 == 0:
            writer.add_scalar('loss',loss.item(), global_step=epoch*len(train_loader)+i)
            print ('Epoch [%d/%d], Batch [%d/%d] Loss: %.6f'
                   %(epoch+1, num_epochs,i+1, len(train_loader),loss.item()))
            _ = validate(i,early_break=True)
    
    pixel_acc,mIoU = validate((epoch+1)*len(train_loader),early_break=False)
    writer.add_scalar('val_pixel_acc',pixel_acc,global_step=(epoch+1)*len(train_loader))
    writer.add_scalar('val_mean_IoU',mIoU,global_step=(epoch+1)*len(train_loader))
    # save Model
    if (not debug):
        try:
            name = opt.model+'_epoch_'+str(epoch+1)+'_model.pkl'
            if opt.use_depth:
                name = opt.model+'_epoch_'+str(epoch+1)+'_use_depth_model.pkl'
            torch.save(model.state_dict(),os.path.join('/home/fangyu/models/',name))
        except:
            print ('save failed.')

# save some logs
"""
writer.export_scalars_to_json("./all_scalars.json")
with open("./all_scalars.json", "a") as myfile:
    myfile.write('\n')
    myfile.write(print (opt))
"""
writer.close()

