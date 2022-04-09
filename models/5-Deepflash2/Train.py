# <h1> HubMap - Hacking the Kidney </h1>
# <h3> Goal - Mapping the human body at function tissue unit level - detect crypt FTUs in colon </h3>
# 
# Implementation of Kaggle Notebook - Innovation Prize Winner - Deep Flash2 <br>
# Description - Train 5 fold model on rescaled images <br>
# Input - train.csv (csv file containing rle format mask), HuBMAP-20-dataset_information.csv (csv containing meta data about the images), Downscaled images and masks, roi-stats.csv (csv containing pdfs for each image), wandb credentials, download deepflash2 library (link - https://www.kaggle.com/matjes/deepflash2-lfs), hubmap_loss_metrics.py <br>
# Output - trained models
# 
# <b>How to use?</b><br> 
# Change the basepath to where your data lives and you're good to go. <br>
# Use the `num_frozen_layers` and `transfer_learning` variables in the Config to turn transfer learning on/off.
# 
# For transfer learning: Set `transfer_learning=True` and `num_frozen_layers=168`. The default number of layers frozen is 168 since it gave the best results, but you can change it to experiment.
# 
# For no transfer learning: Set `transfer_learning=False` and `num_frozen_layers=0`.
# 
# Link to the original notebook -  https://www.kaggle.com/matjes/hubmap-deepflash2-train/data?scriptVersionId=63051354
# 
# 
# <h6> Step 1 - Installation and package loading <h6>
import os
import torch
import zarr
import cv2
import random
import numpy as np
import pandas as pd
from random import shuffle
from scipy import interpolate
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from hubmap_loss_metrics import *
from augmedical.transforms.transforms import ImageTransform
from augmedical.colors.colors import Deconvolution
from tqdm.auto import tqdm
import albumentations as A
import fastai
from fastai.vision.all import *
import segmentation_models_pytorch as smp
from deepflash2.all import *

import wandb
wandb.login(key="6883cb3173ae477ba8d8bde16206f1eaa23dc106")
from fastai.callback.wandb import *

import time 
start = time.time()

def elapsed_time(start_time):
    return time.time() - start_time
# <h6> Step 2 - Make patches of the image file, and define helper functions </h6>

@patch
def apply(self:DeformationField, data, offset=(0, 0), pad=(0, 0), order=1):
    "Apply deformation field to image using interpolation"
    outshape = tuple(int(s - p) for (s, p) in zip(self.shape, pad))
    coords = [np.squeeze(d).astype('float32').reshape(*outshape) for d in self.get(offset, pad)]
    # Get slices to avoid loading all data (.zarr files)
    sl = []
    for i in range(len(coords)):
        cmin, cmax = int(coords[i].min()), int(coords[i].max())
        dmax = data.shape[i]
        if cmin<0: 
            cmax = max(-cmin, cmax)
            cmin = 0 
        elif cmax>dmax:
            cmin = min(cmin, 2*dmax-cmax)
            cmax = dmax
            coords[i] -= cmin
        else: coords[i] -= cmin
        sl.append(slice(cmin, cmax))    
    if len(data.shape) == len(self.shape) + 1:
        tile = np.empty((*outshape, data.shape[-1]))
        for c in range(data.shape[-1]):
            # Adding divide
            tile[..., c] = cv2.remap(data[sl[0],sl[1], c]/255, coords[1],coords[0], interpolation=order, borderMode=cv2.BORDER_REFLECT)
    else:
        tile = cv2.remap(data[sl[0], sl[1]], coords[1], coords[0], interpolation=order, borderMode=cv2.BORDER_REFLECT)
    return tile

class HubmapRandomTileDataset(Dataset):
    """
    Pytorch Dataset that creates random tiles with augmentations from the input images.
    """
    n_inp = 1
    def __init__(self, 
                 files,
                 label_path,
                 cdf_path, 
                 df_stats, 
                 sample_multiplier=50,
                 tile_shape = (512,512),
                 scale = 1,
                 flip = True,                                
                 rotation_range_deg = (0, 360),     
                 deformation_grid = (150,150), 
                 deformation_magnitude = (10,10),
                 value_minimum_range = (0, 0), 
                 value_maximum_range = (1, 1), 
                 value_slope_range = (1, 1),
                 albumentations_tfms=None,
                 augmedical_transforms=None,
                 deconv=True,
                 **kwargs
                ):
        store_attr('files, df_stats, sample_multiplier, tile_shape, scale, albumentations_tfms')
        store_attr('flip, rotation_range_deg, deformation_grid, deformation_magnitude, value_minimum_range, value_maximum_range, value_slope_range')
        
        self.data = zarr.open_group(self.files[0].parent.as_posix(), mode='r')
        self.labels = zarr.open_group(label_path)
        self.cdfs = zarr.open_group(cdf_path)
        
        self.indices = []
        self.center_indices = []
        self.df_stats = self.df_stats[self.df_stats.index.isin([f.stem for f in self.files],  level=0)]
        print('Preparing sampling')
        for key, grp in self.df_stats.groupby('idx'):
            for (idx, i), row in grp.iterrows():
                self.indices.append(idx)
                self.center_indices.append(i)
            for _ in range(self.sample_multiplier):
                self.indices.append(idx)
                self.center_indices.append(None)         
        self.on_epoch_end()
        
        # briefly disable transformations to calc stats
        self.albumentations_tfms = None   
        self.augmedical_transforms = None
        self.deconv = False
        
        if deconv:
            print('Calculating stats for stain normalization w/o albumentation tfms')
            self.dkv_stats = {}
            self.dkv = Deconvolution()
            for f in progress_bar(self.files):
                idxs = [i for i, x in enumerate(self.indices) if x==f.stem]
                t = []
                for i in tqdm(idxs[:100], leave=False):
                    t.append(self[i][0].numpy().transpose(1,2,0))
                
                self.dkv_stats[f.stem] = self.dkv.fit(t)
                
            self.deconv = True
        
            print(self.dkv_stats)
        
        self.albumentations_tfms = albumentations_tfms   
        self.augmedical_transforms = augmedical_transforms
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx): idx = idx.tolist()       
        file_name = self.indices[idx]
        center_idx = self.center_indices[idx]

        img = self.data[file_name]
        n_channels = img.shape[-1]

        lbl = self.labels[file_name]
        cdf = self.cdfs[file_name]

        center = self.random_center(cdf[:], lbl.shape, scale=512, file=file_name, center_idx=center_idx)
        X = self.gammaFcn(self.deformationField.apply(img, center).flatten()).reshape((*self.tile_shape, n_channels))
        Y = self.deformationField.apply(lbl, center, (0,0), 0)

        if self.albumentations_tfms:
            augmented = self.albumentations_tfms(image=(X*255).astype('uint8'),mask=Y.astype('uint8'))
            X = (augmented['image']/255)
            Y = augmented['mask']
            
        if self.deconv:
            d_mean,  d_std = self.dkv_stats[file_name]
            X = self.dkv.apply(X, d_mean, 2*d_std)
            X = np.clip(X, a_min=-5, a_max=5)
            
        X = X.transpose(2, 0, 1).astype('float32')
        Y = Y.astype('int64')
        
        X = TensorImage(X)
        
        if self.augmedical_transforms:
            for transform in self.augmedical_transforms:
                X = transform(X)
        
        return  X, TensorMask(Y)
        
    def random_center(self, cdf, orig_shape, file, center_idx, scale=512):
        'Sample random center'
        if center_idx:
            stats = self.df_stats.loc[file, center_idx]
            cx = random.randrange(stats.top, stats.top+stats.height)
            cy = random.randrange(stats.left, stats.left+stats.width)
        else:
            scale_y = int((orig_shape[1]/orig_shape[0])*scale)
            # print (len(cdf), np.argmax(cdf > np.random.random()), scale, scale_y)
            cx, cy, cz= np.unravel_index(np.argmax(cdf > np.random.random()), (scale,scale_y, 3))
            cx = int(cx*orig_shape[0]/scale)
            cy = int(cy*orig_shape[1]/scale_y)
        return cx, cy
        
    def on_epoch_end(self, verbose=True):

        if verbose: print("Generating deformation field")
        self.deformationField = DeformationField(self.tile_shape, self.scale)

        if self.rotation_range_deg[1] > self.rotation_range_deg[0]:
            self.deformationField.rotate(
                theta=np.pi * (np.random.random()
                            * (self.rotation_range_deg[1] - self.rotation_range_deg[0])
                            + self.rotation_range_deg[0])
                            / 180.0)

        if self.flip:
            self.deformationField.mirror(np.random.choice((True,False),2))

        if self.deformation_grid is not None:
            self.deformationField.addRandomDeformation(
                self.deformation_grid, self.deformation_magnitude)

        if verbose: print("Generating value augmentation function")
        minValue = (self.value_minimum_range[0]
            + (self.value_minimum_range[1] - self.value_minimum_range[0])
            * np.random.random())

        maxValue = (self.value_maximum_range[0]
            + (self.value_maximum_range[1] - self.value_maximum_range[0])
            * np.random.random())

        intermediateValue = 0.5 * (
            self.value_slope_range[0]
            + (self.value_slope_range[1] - self.value_slope_range[0])
            * np.random.random())

        self.gammaFcn = interpolate.interp1d([0, 0.5, 1.0], [minValue, intermediateValue, maxValue], kind="quadratic")  

class HubmapValidationDataset(Dataset):
    "Pytorch Dataset that creates random tiles for validation and prediction on new data."
    n_inp = 1
    def __init__(self, 
                 files, 
                 label_path, 
                 tile_shape = (512,512),
                 scale=1,
                 val_length=None, 
                 val_seed=42, 
                 deconv=True,
                 **kwargs
                ):
        store_attr('files, label_path, tile_shape, scale, val_seed')
        self.data = zarr.open_group(self.files[0].parent.as_posix())
        self.labels = zarr.open_group(label_path)
        self.output_shape = self.tile_shape
        self.tiler = DeformationField(self.tile_shape, scale=self.scale)
        self.image_indices = []
        self.image_shapes = []
        self.centers = []
        self.valid_indices = None

        j = 0
        self.deconv = False
        if deconv: 
            self.dkv = Deconvolution()
            self.dkv_stats = {}
            
        for i, file in enumerate(progress_bar(self.files, leave=False)):
            img = self.data[file.name]
            
            # Tiling
            data_shape = tuple(int(x//self.scale) for x in img.shape[:-1])
            start_points = [o//2 for o in self.output_shape]
            end_points = [(s - st) for s, st in zip(data_shape, start_points)]
            n_points = [int((s)//(o))+1 for s, o in zip(data_shape, self.output_shape)]
            center_points = [np.linspace(st, e, num=n, endpoint=True, dtype=np.int64) for st, e, n in zip(start_points, end_points, n_points)]
            # temp variable for deconv calculation
            image_centers = []
            for cx in center_points[1]:
                for cy in center_points[0]:
                    self.centers.append((int(cy*self.scale), int(cx*self.scale)))
                    image_centers.append((int(cy*self.scale), int(cx*self.scale)))
                    self.image_indices.append(i)
                    self.image_shapes.append(data_shape)
                    j += 1
            
            # Augmedical TFMS
            if deconv:
                count = 0
                t = []
                shuffle(image_centers)
                for center in tqdm(image_centers, leave=False):
                    t.append(self.tiler.apply(img, center))
                
                self.dkv_stats[file.stem] = self.dkv.fit(t)
        
        if deconv: 
            self.deconv = True
            print(self.dkv_stats)
        
        if val_length:
            if val_length>len(self.image_shapes):
                print(f'Reducing validation from lenght {val_length} to {len(self.image_shapes)}')
                val_length = len(self.image_shapes)
            np.random.seed(self.val_seed)
            choice = np.random.choice(len(self.image_indices), val_length, replace=False)
            self.valid_indices = {i:idx for i, idx in  enumerate(choice)}

    def __len__(self):
        if self.valid_indices: return len(self.valid_indices)
        else: return len(self.image_shapes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.valid_indices: idx = self.valid_indices[idx]
        img_path = self.files[self.image_indices[idx]]
        img = self.data[img_path.name]
        centerPos = self.centers[idx]
        X = self.tiler.apply(img, centerPos)
        
        if self.deconv:
            d_mean,  d_std = self.dkv_stats[img_path.name]
            X = self.dkv.apply(X, d_mean, 2*d_std)
            X = np.clip(X, a_min=-5, a_max=5)
            
        X = X.transpose(2, 0, 1).astype('float32')
        
        lbl = self.labels[img_path.name]
        Y = self.tiler.apply(lbl, centerPos, (0,0), order=0).astype('int64')
        
        return  TensorImage(X), TensorMask(Y) 

def show_batch(batch):
    fig, axs = plt.subplots(4,4, figsize=(20,20))   
    images = batch[0].cpu().numpy()
    labels = batch[1].cpu().numpy()

    for i in range(16):     
        axs[i%4, i//4].imshow(images[i, 1])
        axs[i%4, i//4].imshow(labels[i], alpha=0.5)
    plt.show()
    
    plt.hist(batch[0][:,0].cpu().numpy().flatten(), bins=100, alpha=0.5)
    plt.hist(batch[0][:,1].cpu().numpy().flatten(), bins=100, alpha=0.5)
    plt.hist(batch[0][:,2].cpu().numpy().flatten(), bins=100, alpha=0.5)
    plt.show()

dc = TorchLoss(smp.losses.DiceLoss(mode='multiclass', classes=[1]))
ce = CrossEntropyLossFlat(axis=1) #TorchLoss(smp.losses.SoftCrossEntrop

def load_model_weights(model, file, strict=True):
    state = torch.load(file, map_location='cpu')
    stats = state['stats']
    model_state = state['model']
    model.load_state_dict(model_state, strict=strict)
    return model, stats

# from augmentation import Desaturation, GaussianBlur, ChannelBleaching, StainShift
class CONFIG():
    BASE_PATH = r'/N/slate/soodn/'
    dataset = "colon" 
    # dataset = "kidney" 

    INPUT_PATH = BASE_PATH+'hubmap-'+dataset+'-segmentation'
    data_path = Path(f'output_{dataset}/images_scale2')
    annotations_path = Path(f'output_{dataset}/masks_scale2')
    model_path = r'models'
    model_path_transfer_learning = r'models_trained_kidney/unet_model-scale3.pth' 
    
    # deepflash2 dataset
    scale = 1.5 # data is already downscaled to 2, so absolute downscale is 3
    tile_shape = (512, 512)
    sample_multiplier = 100 # Sample 100 tiles from each image, per epoch
    val_length = 500 # Randomly sample 500 validation tiles
    stats = np.array([ 0.0241, -0.0148,  0.0236]), np.array([0.4777, 0.5113, 0.4935]) 
    
    # pytorch model (segmentation_models_pytorch)
    encoder_name = "efficientnet-b2"
    encoder_weights = 'imagenet'
    in_channels = 3
    classes = 2
    
    # Training
    n_splits = 5
    mixed_precision_training = True
    batch_size = 32
    weight_decay = 1e-4 # CHANGED FROM 0.00
    loss_func = JointLoss(dc, ce, 1, 1)
    metrics = [Dice(), Iou(), Recall(), Precision()]
    max_learning_rate = 1e-3
    epochs = 10 # CHANGED FROM 10
    num_frozen_layers = 0 #168
    transfer_learning = False
    
cfg = CONFIG()
# Albumentations augmentations
tfms = A.Compose([
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        ], p=0.3),
    A.OneOf([
        A.Blur(blur_limit=3, p=1),
        A.MedianBlur(blur_limit=3, p=1)
    ], p=.1),
    A.OneOf([
        A.GaussNoise(0.002, p=.5),
        A.IAAAffine(p=.5),
    ], p=.1),
    # Additional position augmentations
    A.RandomRotate90(p=.5),
    A.HorizontalFlip(p=.5),
    A.VerticalFlip(p=.5),
    A.Cutout(num_holes=10,fill_value=255, 
             max_h_size=int(.1 * cfg.tile_shape[0]), 
             max_w_size=int(.1 * cfg.tile_shape[0]), 
             p=.1),
])

# augmedical_transforms = [
#     Desaturation(p=0.0625, max_desaturation=0.25, max_value_reduction=0.25),
#     #Stamping(path="../input/augmentation-images", files=range(1,24), p=cfg.stamping_p, intensity=cfg.stamping_intensity),

#     GaussianBlur(channels=3, p=0.1, kernel_size=3, alpha=0.25),
#     GaussianBlur(channels=3, p=0.0625, kernel_size=23, alpha=0.5),

#     ChannelBleaching(channel=3, p=0.25, min_bleach=0.1, max_bleach=0.25, force_channel=1),
#     ChannelBleaching(channel=3, p=0.0625, min_bleach=0.1, max_bleach=0.5, force_channel=2),
#     ChannelBleaching(channel=3, p=0.0625, min_bleach=0.1, max_bleach=0.5, force_channel=0),

#     #ChannelBlackout(channel=3, p=0.005),
#     StainShift(channel=3, p=0.25, min_shift=1, max_shift=7, force_channel=0),
#     StainShift(channel=3, p=0.25, min_shift=1, max_shift=7, force_channel=2)
# ]


# Position Augmentations
position_augmentation_kwargs = {
    'flip':True,                                
    'rotation_range_deg':(0, 360),     
    'deformation_grid': (150,150), 
    'deformation_magnitude':(10,10),
    'value_minimum_range':(0, 0), 
    'value_maximum_range':(1, 1), 
    'value_slope_range':(1, 1)}

# Datasets
ds_kwargs = {
    'label_path': (cfg.annotations_path/'labels').as_posix(),
    'cdf_path': (cfg.annotations_path/'cdfs').as_posix(),
    'df_stats': pd.read_csv(cfg.annotations_path/'roi_stats.csv', index_col=[0,1]),
    'tile_shape':cfg.tile_shape,
    'scale': cfg.scale,
    'val_length':cfg.val_length, 
    'sample_multiplier':cfg.sample_multiplier,
    'albumentations_tfms': tfms,
   # "augmedical_transforms": augmedical_transforms
}

df_train = pd.read_csv(cfg.INPUT_PATH+'/train.csv')

if cfg.dataset == "colon":
    df_train = df_train.rename(columns={"predicted":"encoding"})
    df_train = df_train[df_train.id != 'HandE_B005_CL_b_RGB_topright']

df_info = pd.read_csv(cfg.INPUT_PATH+'/HuBMAP-20-dataset_information.csv')
files = L([cfg.data_path/x for x in df_train.id])

# <h6> Step 4 - Start k-fold training </h6>
kf = KFold(cfg.n_splits, shuffle=True, random_state=42)
# MODELS = [name for name in glob.glob(cfg.old_mp+'/*.pth')]
for i, (train_idx, val_idx) in enumerate(kf.split(files)):
    files_train, files_val = files[train_idx], files[val_idx]
    print('Training on', [x.name for x in files_train])
    
    # Datasets
    train_ds = HubmapRandomTileDataset(files_train, **ds_kwargs, **position_augmentation_kwargs)
    valid_ds = HubmapValidationDataset(files_val, **ds_kwargs)
    
    # Model
    model = smp.Unet(encoder_name=cfg.encoder_name, 
                     encoder_weights=cfg.encoder_weights, 
                     in_channels=cfg.in_channels, 
                     classes=cfg.classes)
    
    if cfg.transfer_learning:
        print("Transfer learning: True")
        m_path = cfg.model_path_transfer_learning
        model, stats = load_model_weights(model, m_path)

    val = 0
    for name, param in model.named_parameters():
        if val == cfg.num_frozen_layers:
            break    
        else:
            param.requires_grad = False
        val+=1
    
    count_frozen = 0
    count_unfrozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            count_frozen +=1 
        else:
            count_unfrozen += 1
    
    print(f"Number of frozen layers: {count_frozen}")
    print(f"Number of unfrozen layers: {count_unfrozen}")
    
    # Dataloader and learner
    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=cfg.batch_size, after_batch=Normalize.from_stats(*cfg.stats))
    if torch.cuda.is_available(): dls.cuda(), model.cuda()
    
    if i==0: 
        show_batch(dls.one_batch())
        
    run = wandb.init(project='bricknet', reinit=True, config=cfg, name=f"default_with_phils_augment_{i}")

    cbs = [SaveModelCallback(monitor='dice'), ElasticDeformCallback, WandbCallback(log_preds=False, log_model=False)]
    learn = Learner(dls, model, metrics=cfg.metrics, wd=cfg.weight_decay, loss_func=cfg.loss_func, opt_func=ranger, cbs=cbs)
    if cfg.mixed_precision_training: learn.to_fp16()
    
    print ("Start model fitting", learn)
    # Fit
    learn.fit_one_cycle(cfg.epochs, lr_max=cfg.max_learning_rate)
    learn.recorder.plot_metrics()
    
    # Save Model
    print ("Saving Model")
    state = {'model': learn.model.state_dict(), 'stats':cfg.stats}
    torch.save(state, f'models_kidney/unet_{cfg.encoder_name}_{i}.pth', pickle_protocol=2, _use_new_zipfile_serialization=False)

print ("Run time = ", elapsed_time(start))