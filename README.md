# PyTorch Semantic Segmentation Zoo
- Currently under construction.


## run
```
python3 train.py --train_rgb_path ./data/image/train \
                 --train_depth_path ./data/depth/train \
                 --train_label_path ./data/label37/train \
                 --val_rgb_path ./data/image/test \
                 --val_depth_path ./data/depth/test \
                 --val_label_path ./data/label37/test \
                 --batch_size 2 \
                 --lr 0.001 \
                 --n_classes 38 \
                 --model Deeplab-v2 \
                 --data SUNRGBD \
                 --log_path runs/test \
                 --epochs 30
```


##  Requirements
- PyTorch >= 4.0
- tensorboardX

##  Supported models
(Most models are referenced from open source projects. Please check top of[model].py for source(s).)
- U-Net
- ResNet-DUC-HDC
- DeepLab-v2 (ResNet)
- DeepLab-LargeFOV (VGG16)

##  Supported datasets
- SUN RGB-D
- Pascal VOC 2012 

## TODO
- `tensorboardX` for visualizing logs
- `inference.py`
