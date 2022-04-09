DEBUG = False

import numpy as np
import pandas as pd
pd.get_option("display.max_columns")
pd.set_option('display.max_columns', 300)
pd.get_option("display.max_rows")
pd.set_option('display.max_rows', 300)

import matplotlib.pyplot as plt

import sys
import os
from os.path import join as opj
import random
import copy

import cv2
import rasterio
from rasterio.windows import Window
import tifffile
from skimage import measure

import time
from tqdm.notebook import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)
from albumentations.pytorch import ToTensorV2

package_dir = "pretrainedmodels/pretrained-models.pytorch-master/"
sys.path.insert(0, package_dir)
import pretrainedmodels

BASE_PATH = r'/N/slate/soodn/'

config = {
    'split_seed_list':[0],
    'FOLD_LIST':[0,1,2,3], 
    'model_path':BASE_PATH+'models/hubmap-new-03-03',
    'model_name':'seresnext101',
    
    'num_classes':1,
    'resolution':1024, #(1024,1024),(512,512),
    'input_resolution':320, #(320,320), #(256,256), #(512,512), #(384,384)
    'deepsupervision':False, # always false for inference
    'clfhead':False,
    'clf_threshold':0.5,
    'small_mask_threshold':0, #256*256*0.03, #512*512*0.03,
    'mask_threshold':0.5,
    # 'mask_threshold':0.0003,
    'pad_size':256, #(64,64), #(256,256), #(128,128)
    
    'tta':3,
    'test_batch_size':12,
    
    'FP16':False,
    'num_workers':4,
    'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

device = config['device']
start = time.time()

# dataset = "colon" 
# dataset = "kidney" 
dataset = "new-data"

INPUT_PATH = BASE_PATH+'hubmap-'+dataset+'-segmentation'

print('Python        : ' + sys.version.split('\n')[0])
print('Numpy         : ' + np.__version__)
print('Pandas        : ' + pd.__version__)
print('Rasterio      : ' + rasterio.__version__)
print('OpenCV        : ' + cv2.__version__)

train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
info_df  = pd.read_csv(opj(INPUT_PATH,'HuBMAP-20-dataset_information.csv'))
sub_df = pd.read_csv(opj(INPUT_PATH, 'sample_submission.csv'))

print('train_df.shape = ', train_df.shape)
print('info_df.shape  = ', info_df.shape)
print('sub_df.shape = ', sub_df.shape)

if len(sub_df) == 5:
    if DEBUG:
        sub_df = sub_df[:1]
    else:
        sub_df = sub_df[:]

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
fix_seed(2020)
    
def elapsed_time(start_time):
    return time.time() - start_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rle2mask(rle, shape):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction

def mask2rle(img, shape, small_mask_threshold):
    if img.shape != shape:
        h,w = shape
        img = cv2.resize(img, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.int8) 
    pixels = img.T.flatten()
    #pixels = np.concatenate([[0], pixels, [0]])
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if runs[1::2].sum() <= small_mask_threshold:
        return ''
    else:
        return ' '.join(str(x) for x in runs)

def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform_(m.weight, gain=1)
        #nn.init.xavier_normal_(m.weight, gain=1)
        #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

class cSEBlock(nn.Module):
    def __init__(self, c, feat):
        super().__init__()
        self.attention_fc = nn.Linear(feat,1, bias=False)
        self.bias         = nn.Parameter(torch.zeros((1,c,1), requires_grad=True))
        self.sigmoid      = nn.Sigmoid()
        self.dropout      = nn.Dropout2d(0.1)
        
    def forward(self,inputs):
        batch,c,h,w = inputs.size()
        x = inputs.view(batch,c,-1)
        x = self.attention_fc(x) + self.bias
        x = x.view(batch,c,1,1)
        x = self.sigmoid(x)
        x = self.dropout(x)
        return inputs * x

class sSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.attention_fc = nn.Linear(c,1, bias=False).apply(init_weight)
        self.bias         = nn.Parameter(torch.zeros((1,h,w,1), requires_grad=True))
        self.sigmoid      = nn.Sigmoid()
        
    def forward(self,inputs):
        batch,c,h,w = inputs.size()
        x = torch.transpose(inputs, 1,2) #(*,c,h,w)->(*,h,c,w)
        x = torch.transpose(x, 2,3) #(*,h,c,w)->(*,h,w,c)
        x = self.attention_fc(x) + self.bias
        x = torch.transpose(x, 2,3) #(*,h,w,1)->(*,h,1,w)
        x = torch.transpose(x, 1,2) #(*,h,1,w)->(*,1,h,w)
        x = self.sigmoid(x)
        return inputs * x
    
class scSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.cSE = cSEBlock(c,h*w)
        self.sSE = sSEBlock(c,h,w)
    
    def forward(self, inputs):
        x1 = self.cSE(inputs)
        x2 = self.sSE(inputs)
        return x1+x2
    
class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta    = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.phi      = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.g        = nn.utils.spectral_norm(conv1x1(channels, channels//2)).apply(init_weight)
        self.o        = nn.utils.spectral_norm(conv1x1(channels//2, channels)).apply(init_weight)
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel//reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel//reduction, in_channel).apply(init_weight)
        )
        
    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x  = torch.sigmoid(x1 + x2)
        return x
    
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2,1).apply(init_weight)
        
    def forward(self, inputs):
        x1,_ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x  = torch.cat([x1,x2], dim=1)
        x  = self.conv3x3(x)
        x  = torch.sigmoid(x)
        return x
    
class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x
    
class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1   = conv1x1(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x  = F.relu(self.bn1(inputs))
        x  = self.upsample(x)
        x  = self.conv3x3_1(x)
        x  = self.conv3x3_2(F.relu(self.bn2(x)))
        x  = self.cbam(x)
        x += self.conv1x1(self.upsample(inputs)) #shortcut
        return x
    
#U-Net ResNet34 + CBAM + hypercolumns + deepsupervision
class UNET_RESNET34(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h,w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        
        #encoder
        model_name = 'resnet34' #26M
        resnet34 = pretrainedmodels.__dict__['resnet34'](num_classes=1000,pretrained=None)
        if load_weights:
            resnet34.load_state_dict(torch.load(f'../../../pretrainedmodels_weight/{model_name}.pth'))
        self.conv1   = resnet34.conv1 #(*,3,h,w)->(*,64,h/2,w/2)
        self.bn1     = resnet34.bn1
        self.maxpool = resnet34.maxpool #->(*,64,h/4,w/4)
        self.layer1  = resnet34.layer1 #->(*,64,h/4,w/4) 
        self.layer2  = resnet34.layer2 #->(*,128,h/8,w/8) 
        self.layer3  = resnet34.layer3 #->(*,256,h/16,w/16) 
        self.layer4  = resnet34.layer4 #->(*,512,h/32,w/32) 
        
        #center
        self.center  = CenterBlock(512,512) #->(*,512,h/32,w/32) 
        
        #decoder
        self.decoder4 = DecodeBlock(512+512,64, upsample=True) #->(*,64,h/16,w/16) 
        self.decoder3 = DecodeBlock(64+256,64, upsample=True) #->(*,64,h/8,w/8) 
        self.decoder2 = DecodeBlock(64+128,64,  upsample=True) #->(*,64,h/4,w/4) 
        self.decoder1 = DecodeBlock(64+64,64,   upsample=True) #->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #deep supervision
        self.deep4 = conv1x1(64,1).apply(init_weight)
        self.deep3 = conv1x1(64,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(64,1).apply(init_weight)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,1).apply(init_weight)
        )
        
        #clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,1).apply(init_weight)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = F.relu(self.bn1(self.conv1(inputs))) #->(*,64,h/2,w/2) 
        x0 = self.maxpool(x0) #->(*,64,h/4,w/4)
        x1 = self.layer1(x0) #->(*,64,h/4,w/4)
        x2 = self.layer2(x1) #->(*,128,h/8,w/8)
        x3 = self.layer3(x2) #->(*,256,h/16,w/16)
        x4 = self.layer4(x3) #->(*,512,h/32,w/32)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        if config['clf_threshold'] is not None:
            if (torch.sigmoid(logits_clf)>config['clf_threshold']).sum().item()==0:
                bs,_,h,w = inputs.shape
                logits = torch.zeros((bs,1,h,w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits,_,_
                    else:
                        return logits,_
                else:
                    if self.deepsupervision:
                        return logits,_
                    else:
                        return logits
        
        #center
        y5 = self.center(x4) #->(*,512,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2)
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,1,h,w)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        
        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps
            else:
                return logits
        
#U-Net SeResNext50 + CBAM + hypercolumns + deepsupervision
class UNET_SERESNEXT50(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h,w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        
        #encoder
        model_name = 'se_resnext50_32x4d' #26M
        seresnext50 = pretrainedmodels.__dict__[model_name](pretrained=None)
        if load_weights:
            seresnext50.load_state_dict(torch.load(f'../../../pretrainedmodels_weight/{model_name}.pth'))
        
        self.encoder0 = nn.Sequential(
            seresnext50.layer0.conv1, #(*,3,h,w)->(*,64,h/2,w/2)
            seresnext50.layer0.bn1,
            seresnext50.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext50.layer0.pool, #->(*,64,h/4,w/4)
            seresnext50.layer1 #->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext50.layer2 #->(*,512,h/8,w/8)
        self.encoder3 = seresnext50.layer3 #->(*,1024,h/16,w/16)
        self.encoder4 = seresnext50.layer4 #->(*,2048,h/32,w/32)
        
        #center
        self.center  = CenterBlock(2048,512) #->(*,512,h/32,w/32) 10,16
        
        #decoder
        self.decoder4 = DecodeBlock(512+2048,64, upsample=True) #->(*,64,h/16,w/16) 20,32
        self.decoder3 = DecodeBlock(64+1024,64, upsample=True) #->(*,64,h/8,w/8) 40,64
        self.decoder2 = DecodeBlock(64+512,64,  upsample=True) #->(*,64,h/4,w/4) 80,128
        self.decoder1 = DecodeBlock(64+256,64,   upsample=True) #->(*,64,h/2,w/2) 160,256
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 320,512
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #deep supervision
        self.deep4 = conv1x1(64,1).apply(init_weight)
        self.deep3 = conv1x1(64,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(64,1).apply(init_weight)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,1).apply(init_weight)
        )
        
        #clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(2048).apply(init_weight),
            nn.Linear(2048,512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,1).apply(init_weight)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = self.encoder0(inputs) #->(*,64,h/2,w/2) 160,256
        x1 = self.encoder1(x0) #->(*,256,h/4,w/4)
        x2 = self.encoder2(x1) #->(*,512,h/8,w/8)
        x3 = self.encoder3(x2) #->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3) #->(*,2048,h/32,w/32)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        if config['clf_threshold'] is not None:
            if (torch.sigmoid(logits_clf)>config['clf_threshold']).sum().item()==0:
                bs,_,h,w = inputs.shape
                logits = torch.zeros((bs,1,h,w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits,_,_
                    else:
                        return logits,_
                else:
                    if self.deepsupervision:
                        return logits,_
                    else:
                        return logits
        
        #center
        y5 = self.center(x4) #->(*,320,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2) 160,256
        y0 = self.decoder0(y1) #->(*,64,h,w) 320,512
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,4,h,w)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        
        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps
            else:
                return logits

#U-Net SeResNext101 + CBAM + hypercolumns + deepsupervision
class UNET_SERESNEXT101(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h,w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        
        #encoder
        model_name = 'se_resnext101_32x4d'
        seresnext101 = pretrainedmodels.__dict__[model_name](pretrained=None)
        if load_weights:
            seresnext101.load_state_dict(torch.load(f'../../../pretrainedmodels_weight/{model_name}.pth'))
        
        self.encoder0 = nn.Sequential(
            seresnext101.layer0.conv1, #(*,3,h,w)->(*,64,h/2,w/2)
            seresnext101.layer0.bn1,
            seresnext101.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext101.layer0.pool, #->(*,64,h/4,w/4)
            seresnext101.layer1 #->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext101.layer2 #->(*,512,h/8,w/8)
        self.encoder3 = seresnext101.layer3 #->(*,1024,h/16,w/16)
        self.encoder4 = seresnext101.layer4 #->(*,2048,h/32,w/32)
        
        #center
        self.center  = CenterBlock(2048,512) #->(*,512,h/32,w/32)
        
        #decoder
        self.decoder4 = DecodeBlock(512+2048,64, upsample=True) #->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(64+1024,64, upsample=True) #->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(64+512,64,  upsample=True) #->(*,64,h/4,w/4) 
        self.decoder1 = DecodeBlock(64+256,64,   upsample=True) #->(*,64,h/2,w/2) 
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #deep supervision
        self.deep4 = conv1x1(64,1).apply(init_weight)
        self.deep3 = conv1x1(64,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(64,1).apply(init_weight)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,1).apply(init_weight)
        )
        
        #clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(2048).apply(init_weight),
            nn.Linear(2048,512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,1).apply(init_weight)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = self.encoder0(inputs) #->(*,64,h/2,w/2)
        x1 = self.encoder1(x0) #->(*,256,h/4,w/4)
        x2 = self.encoder2(x1) #->(*,512,h/8,w/8)
        x3 = self.encoder3(x2) #->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3) #->(*,2048,h/32,w/32)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        if config['clf_threshold'] is not None:
            if (torch.sigmoid(logits_clf)>config['clf_threshold']).sum().item()==0:
                bs,_,h,w = inputs.shape
                logits = torch.zeros((bs,1,h,w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits,_,_
                    else:
                        return logits,_
                else:
                    if self.deepsupervision:
                        return logits,_
                    else:
                        return logits
        
        #center
        y5 = self.center(x4) #->(*,320,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2) 
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,1,h,w)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        
        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps
            else:
                return logits    
  
def build_model(resolution, deepsupervision, clfhead, load_weights):
    model_name = config['model_name']
    if model_name=='resnet34':
        model = UNET_RESNET34(resolution, deepsupervision, clfhead, load_weights)
    elif model_name=='seresnext50':
        model = UNET_SERESNEXT50(resolution, deepsupervision, clfhead, load_weights)
    elif model_name=='seresnext101':
        model = UNET_SERESNEXT101(resolution, deepsupervision, clfhead, load_weights)
    return model

LOAD_LOCAL_WEIGHT_PATH_LIST = {}
for seed in config['split_seed_list']:
    LOAD_LOCAL_WEIGHT_PATH_LIST[seed] = []
    for fold in config['FOLD_LIST']:
        LOAD_LOCAL_WEIGHT_PATH_LIST[seed].append(opj(config['model_path'],f'model_seed{seed}_fold{fold}_bestscore.pth'))
        #LOAD_LOCAL_WEIGHT_PATH_LIST[seed].append(opj(config['model_path'],f'model_seed{seed}_fold{fold}_swa.pth'))

model_list = {}
for seed in config['split_seed_list']:
    model_list[seed] = []
    for path in LOAD_LOCAL_WEIGHT_PATH_LIST[seed]:
        print("Loading weights from %s" % path)
        
        model = build_model(resolution=(None,None), #config['resolution'], 
                            deepsupervision=config['deepsupervision'], 
                            clfhead=config['clfhead'],
                            load_weights=False).to(device)
        
        model.load_state_dict(torch.load(path))
        model.eval()
        model_list[seed].append(model) 


#from get_config import *
#config = get_config()

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def get_transforms_test():
    transforms = Compose([
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
        ToTensorV2(),
    ] )
    return transforms

def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean


def load_image(img_path, df_info):
    """
    Load image and make sure sizes matches df_info
    """
    fname = img_path.rsplit("/", -1)[-1]
    image_fname = fname.rsplit(".")[0]
    W = int(df_info[df_info.image_file == image_fname]["width_pixels"])
    H = int(df_info[df_info.image_file == image_fname]["height_pixels"])

    img = tifffile.imread(img_path).squeeze()

    channel_pos = np.argwhere(np.array(img.shape) == 3)[0][0]
    W_pos = np.argwhere(np.array(img.shape) == W)[0][0]
    H_pos = np.argwhere(np.array(img.shape) == H)[0][0]

    img = np.moveaxis(img, (H_pos, W_pos, channel_pos), (0, 1, 2))
    return img

class HuBMAPDataset(Dataset):
    def __init__(self, idx, df):
        super().__init__()
        filename = df.loc[idx, 'id']+'.tiff'
        IMAGE_PATH = INPUT_PATH+'/test'
        path = opj(IMAGE_PATH,filename)
        self.data = rasterio.open(path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.input_sz = config['input_resolution']
        self.sz = config['resolution']
        self.pad_sz = config['pad_size'] # add to each input tile
        self.pred_sz = self.sz - 2*self.pad_sz
        self.pad_h = self.pred_sz - self.h % self.pred_sz # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz
        self.transforms = get_transforms_test()
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx):
        # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.pred_sz 
        x = i_w*self.pred_sz
        py0,py1 = max(0,y), min(y+self.pred_sz, self.h)
        px0,px1 = max(0,x), min(x+self.pred_sz, self.w)
        
        # padding coordinate for rasterio
        qy0,qy1 = max(0,y-self.pad_sz), min(y+self.pred_sz+self.pad_sz, self.h)
        qx0,qx1 = max(0,x-self.pad_sz), min(x+self.pred_sz+self.pad_sz, self.w)
        
        # placeholder for input tile (before resize)
        img = np.zeros((self.sz,self.sz,3), np.uint8)
        
        # replace the value
        if self.data.count == 3: 
            img[0:qy1-qy0, 0:qx1-qx0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((qy0,qy1),(qx0,qx1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img[0:qy1-qy0, 0:qx1-qx0, i] =\
                    layer.read(1,window=Window.from_slices((qy0,qy1),(qx0,qx1)))
        if self.sz != self.input_sz:
            img = cv2.resize(img, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)
        img = self.transforms(image=img)['image'] # to normalized tensor
        return {'img':img, 'p':[py0,py1,px0,px1], 'q':[qy0,qy1,qx0,qx1]}

def my_collate_fn(batch):
    img = []
    p = []
    q = []
    for sample in batch:
        img.append(sample['img'])
        p.append(sample['p'])
        q.append(sample['q'])
    img = torch.stack(img)
    return {'img':img, 'p':p, 'q':q}


seed = 0
def get_pred_mask(idx, df, model_list):
    ds = HuBMAPDataset(idx, df)
    #rasterio cannot be used with multiple workers
    dl = DataLoader(ds,batch_size=config['test_batch_size'],
                    num_workers=0,shuffle=False,pin_memory=True,
                    collate_fn=my_collate_fn) 
    
    pred_mask = np.zeros((len(ds),ds.pred_sz,ds.pred_sz), dtype=np.uint8)
    
    i_data = 0
    for data in tqdm(dl):
        bs = data['img'].shape[0]
        img_patch = data['img'] # (bs,3,input_res,input_res)
        pred_mask_float = 0
        for model in model_list[seed]:
            with torch.no_grad():
                if config['tta']>0:
                    pred_mask_float += torch.sigmoid(model(img_patch.to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                if config['tta']>1:
                    # h-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-1]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,:,::-1]
                if config['tta']>2:
                    # v-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-2]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,::-1,:]
                if config['tta']>3:
                    # h-v-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-1,-2]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,::-1,::-1]
        pred_mask_float = pred_mask_float / min(config['tta'],4) / len(model_list[seed]) # (bs,input_res,input_res)
        
        # resize
        pred_mask_float = np.vstack([cv2.resize(_mask.astype(np.float32), (ds.sz,ds.sz))[None] for _mask in pred_mask_float])
        
        # float to uint8
        pred_mask_int = (pred_mask_float>config['mask_threshold']).astype(np.uint8)
        # replace the values
        for j in range(bs):
            py0,py1,px0,px1 = data['p'][j]
            qy0,qy1,qx0,qx1 = data['q'][j]
            pred_mask[i_data+j,0:py1-py0, 0:px1-px0] = pred_mask_int[j, py0-qy0:py1-qy0, px0-qx0:px1-qx0] # (pred_sz,pred_sz)
        i_data += bs
    
    pred_mask = pred_mask.reshape(ds.num_h*ds.num_w, ds.pred_sz, ds.pred_sz).reshape(ds.num_h, ds.num_w, ds.pred_sz, ds.pred_sz)
    pred_mask = pred_mask.transpose(0,2,1,3).reshape(ds.num_h*ds.pred_sz, ds.num_w*ds.pred_sz)
    pred_mask = pred_mask[:ds.h,:ds.w] # back to the original slide size
    non_zero_ratio = (pred_mask).sum() / (ds.h*ds.w)
    print('non_zero_ratio = {:.4f}'.format(non_zero_ratio))
    return pred_mask,ds.h,ds.w

def get_rle(y_preds, h,w):
    rle = mask2rle(y_preds, shape=(h,w), small_mask_threshold=config['small_mask_threshold'])
    return rle

def mask2json(mask):
    contours = measure.find_contours(mask, 0.8)
    # contour to polygon
    polygons = []
    for object in contours:
        coords = []
        for point in object:
            coords.append([int(point[1]), int(point[0])])
        polygons.append(coords)
    # save as json
    geojson_dict_template = {
        "type": "Feature",
        "id": "PathAnnotationObject",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
            ]
        },
        "properties": {
            "classification": {
                "name": "glomerulus",
                "colorRGB": -3140401
            },
            "isLocked": True,
            "measurements": []
        }
    }
    geojson_list = []
    for polygon in polygons:
        geojson_dict = copy.deepcopy(geojson_dict_template)
        geojson_dict["geometry"]["coordinates"].append(polygon)
        geojson_list.append(geojson_dict)

    return geojson_list

for idx in range(0,len(sub_df)): 
    pred_mask, h, w = get_pred_mask(idx, sub_df, model_list)  
    rle = get_rle(pred_mask,h,w)
    print ("RLE:", rle)
    sub_df.loc[idx,'predicted'] = rle
    plt.imsave(f'mask_{idx}.png', pred_mask)

sub_df.to_csv('submission-'+dataset+".csv", index=False)
print ("Run time = ", elapsed_time(start))