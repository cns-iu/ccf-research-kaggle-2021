# <h1> HubMap - Hacking the Kidney </h1>
# <h3> Goal - Mapping the human body at function tissue unit level - detect crypts FTUs in colon </h3>
# 
# Implementation of Kaggle Notebook - Scientific Prize Winner - Deeplive <br>
# Description - Use 2-5 fold models to predict on test image masks <br>
# Input - models, test images, sample_submission.csv  <br>
# Output - submission_deeplive_generalized.csv (rle for test images) <br>
# 
# <b>How to use?</b><br> 
# Change the basepath to where your data lives and you're good to go. <br>
# 
# Link 1 - kaggle.com/optimo/hubmap-inference-th-o 
# 
# <h6> Step 1 - Import useful libraries </h6>
import os
import gc
import cv2
import json
import glob
import torch
import random
import rasterio
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
import segmentation_models_pytorch
from segmentation_models_pytorch.encoders import encoders
from tqdm.notebook import tqdm
from collections import Counter
from rasterio.windows import Window
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

import albumentations as albu
from albumentations.pytorch import ToTensorV2

import time
start = time.time()

def elapsed_time(start_time):
    return time.time() - start_time

# <h6> Step 2 - Helper Functions </h6>
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
SEED = 2021
seed_everything(SEED)
CLASSES = ["ftus"]
NUM_CLASSES = len(CLASSES)
MEAN = np.array([0.66437738, 0.50478148, 0.70114894])
STD = np.array([0.15825711, 0.24371008, 0.13832686])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDENTITY = rasterio.Affine(1, 0, 0, 0, 1, 0)
FLIPS = [[-1], [-2], [-2, -1]]

def load_image(img_path, df_info, reduce_factor=1):
    """
    Load image and make sure sizes matches df_info
    """
    image_fname = img_path.rsplit("/", -1)[-1]
    
    
    W = int(df_info[df_info.image_file == image_fname]["width_pixels"])
    H = int(df_info[df_info.image_file == image_fname]["height_pixels"])

    img = tiff.imread(img_path).squeeze()

    channel_pos = np.argwhere(np.array(img.shape) == 3)[0][0]
    W_pos = np.argwhere(np.array(img.shape) == W)[0][0]
    H_pos = np.argwhere(np.array(img.shape) == H)[0][0]

    img = np.moveaxis(img, (H_pos, W_pos, channel_pos), (0, 1, 2))
    
    if reduce_factor > 1:
        img = cv2.resize(
            img,
            (img.shape[1] // reduce_factor, img.shape[0] // reduce_factor),
            interpolation=cv2.INTER_AREA,
        )
        
    return img

def HE_preprocess(augment=False, visualize=False, mean=MEAN, std=STD):
    """
    Returns transformations for the H&E images.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.
        mean ([type], optional): Mean for normalization. Defaults to MEAN.
        std ([type], optional): Standard deviation for normalization. Defaults to STD.

    Returns:
        albumentation transforms: transforms.
    """
    if visualize:
        normalizer = albu.Compose(
            [albu.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), ToTensorV2()], p=1
        )
    else:
        normalizer = albu.Compose(
            [albu.Normalize(mean=mean, std=std), ToTensorV2()], p=1
        )
    
    if augment:
        raise NotImplementedError

    return normalizer

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        enc_split = enc.split()
        for i in range(len(enc_split) // 2):
            start = int(enc_split[2 * i]) - 1
            length = int(enc_split[2 * i + 1])
            img[start: start + length] = 1 + m

    return img.reshape(shape).T

class InferenceDataset(Dataset):
    def __init__(
        self,
        original_img_path,
        rle=None,
        overlap_factor=1,
        tile_size=256,
        reduce_factor=4,
        transforms=None,
    ):
        self.original_img = load_image(original_img_path, full_size=reduce_factor > 1)
        self.orig_size = self.original_img.shape

        # self.original_img = lab_normalization(self.original_img)

        self.raw_tile_size = tile_size
        self.reduce_factor = reduce_factor
        self.tile_size = tile_size * reduce_factor

        self.overlap_factor = overlap_factor

        self.positions = self.get_positions()
        self.transforms = transforms

        if rle is not None:
            self.mask = enc2mask(rle, (self.orig_size[1], self.orig_size[0])) > 0
        else:
            self.mask = None

    def __len__(self):
        return len(self.positions)

    def get_positions(self):
        top_x = np.arange(
            0,
            self.orig_size[0],  # +self.tile_size,
            int(self.tile_size / self.overlap_factor),
        )
        top_y = np.arange(
            0,
            self.orig_size[1],  # +self.tile_size,
            int(self.tile_size / self.overlap_factor),
        )
        starting_positions = []
        for x in top_x:
            right_space = self.orig_size[0] - (x + self.tile_size)
            if right_space > 0:
                boundaries_x = (x, x + self.tile_size)
            else:
                boundaries_x = (x + right_space, x + right_space + self.tile_size)

            for y in top_y:
                down_space = self.orig_size[1] - (y + self.tile_size)
                if down_space > 0:
                    boundaries_y = (y, y + self.tile_size)
                else:
                    boundaries_y = (y + down_space, y + down_space + self.tile_size)
                starting_positions.append((boundaries_x, boundaries_y))

        return starting_positions

    def __getitem__(self, idx):
        pos_x, pos_y = self.positions[idx]
        img = self.original_img[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1], :]

        # img = lab_normalization(img)

        # down scale to tile size
        if self.reduce_factor > 1:
            img = cv2.resize(
                img, (self.raw_tile_size, self.raw_tile_size), interpolation=cv2.INTER_AREA
            )

        if self.transforms:
            img = self.transforms(image=img)["image"]

        pos = np.array([pos_x[0], pos_x[1], pos_y[0], pos_y[1]])

        return img, pos
    
class InferenceEfficientDataset(InferenceDataset):
    """
    Refs : 
    https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter-sub/
    https://www.kaggle.com/finlay/pytorch-fcn-resnet50-in-20-minute
    """
    def __init__(
        self,
        original_img_path,
        rle=None,
        overlap_factor=1,
        tile_size=256,
        reduce_factor=4,
        transforms=None,
    ):
            
        self.raw_tile_size = tile_size
        self.reduce_factor = reduce_factor
        self.tile_size = tile_size * reduce_factor
        
        self.overlap_factor = overlap_factor
        self.transforms = transforms

        # Load image with rasterio        
        self.original_img = rasterio.open(original_img_path, transform=IDENTITY, num_threads='all_cpus')
        if self.original_img.count != 3:
            self.layers = [rasterio.open(subd) for subd in self.original_img.subdatasets]
                    
        self.orig_size = self.original_img.shape

        self.positions = self.get_positions()
        
        if rle is not None:
            self.mask = enc2mask(rle, (self.orig_size[1], self.orig_size[0])) > 0
        else:
            self.mask = None

    def __getitem__(self, idx):
        
        # Window
        pos_x, pos_y = self.positions[idx]
        x1, x2 = pos_x[0], pos_x[1]
        y1, y2 = pos_y[0], pos_y[1]
        window = Window.from_slices((x1, x2), (y1, y2))

        # Retrieve slice
        if self.original_img.count == 3:  # normal
            img = self.original_img.read([1, 2, 3], window=window)
            img = np.moveaxis(img, 0, -1)
        else:  # with subdatasets/layers
            img = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
            for fl in range(3):
                img[:, :, fl] = self.layers[fl].read(window=window) 

        # Downscale to tile size
        img = cv2.resize(
            img, (self.raw_tile_size, self.raw_tile_size), interpolation=cv2.INTER_AREA
        )
        img = self.transforms(image=img)["image"]
        
        pos = np.array([pos_x[0], pos_x[1], pos_y[0], pos_y[1]])

        return img, pos

def load_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".

    Returns:
        torch model: Model with loaded weights.
    """

    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=True)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )
    return model

DECODERS = ["Unet", "Linknet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "PAN"]
ENCODERS = list(encoders.keys())

def define_model(
    decoder_name, encoder_name, num_classes=1, activation=None, encoder_weights="imagenet"
):
    """
    Loads a segmentation architecture

    Args:
        decoder_name (str): Decoder name.
        encoder_name (str): Encoder name.
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained : pretrained original weights
    Returns:
        torch model -- Pretrained model.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"
    assert encoder_name in ENCODERS, "Encoder name not supported"

    decoder = getattr(segmentation_models_pytorch, decoder_name)

    model = decoder(
        encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation,
    )
    model.num_classes = num_classes

    return model

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_models(cp_folder):
    config = json.load(open(cp_folder+'/config.json', 'r'))
    config = Config(**config)
    
    weights = sorted(glob.glob(str(cp_folder+'/*.pt')))
    models = []
    
    for weight in weights:
        model = define_model(
            config.decoder,
            config.encoder,
            num_classes=config.num_classes,
            encoder_weights=None,
        )
        
        model = load_model_weights(model, weight).to(DEVICE)
        model.eval()
        models.append(model)        
    return models

def dice_scores_img(pred, truth, eps=1e-8):
    """
    Dice metric for a single image as array.

    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.

    Returns:
        np array : dice value for each class
    """
    pred = pred.reshape(-1) > 0
    truth = truth.reshape(-1) > 0
    intersect = (pred & truth).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return dice

def get_tile_weighting(size, sigma=1, alpha=1, eps=1e-6):
    half = size // 2
    w = np.ones((size, size), np.float32)

    x = np.concatenate([np.mgrid[-half:0], np.mgrid[1: half + 1]])[:, None]
    x = np.tile(x, (1, size))
    x = half + 1 - np.abs(x)
    y = x.T

    w = np.minimum(x, y)
    w = (w / w.max()) ** sigma
    w = np.minimum(w, 1)

    w = (w - np.min(w) + eps) / (np.max(w) - np.min(w) + eps)

    w = np.where(w > alpha, 1, w)
    w = w / alpha
    w = np.clip(w, 1e-3, 1)

    w = np.round(w, 3)
    return w.astype(np.float16)

def predict_entire_mask(dataset, models, batch_size=32, tta=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    weighting = torch.from_numpy(get_tile_weighting(dataset.tile_size, sigma=1, alpha=1))
    weighting_cuda = weighting.clone().cuda().unsqueeze(0)
    weighting = weighting.cuda().half()

    global_pred = torch.zeros(
        (dataset.orig_size[0], dataset.orig_size[1]),
        dtype=torch.half, device="cuda"
    )
    global_counter = torch.zeros(
        (dataset.orig_size[0], dataset.orig_size[1]),
        dtype=torch.half, device="cuda"
    )

    with torch.no_grad():
        for img, pos in tqdm(loader):
            img = img.to("cuda")
            _, _, h, w = img.shape
            
            model_preds = []
            for model in models:
                if model.num_classes == 1:
                    pred = model(img).view(1, -1, h, w).sigmoid().detach()
                else:
                    pred = model(img)[:, 0].view(1, -1, h, w).sigmoid().detach()

                if tta:
                    for f in FLIPS:
                        pred_flip = model(torch.flip(img, f))
                        if model.num_classes == 2:
                            pred_flip = pred_flip[:, :1]

                        pred_flip = torch.flip(pred_flip, f).view(1, -1, h, w).sigmoid().detach()
                        pred += pred_flip
                    pred = torch.div(pred, len(FLIPS) + 1)

                model_preds.append(pred)

            pred = torch.cat(model_preds, 0).mean(0)

            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), (dataset.tile_size, dataset.tile_size), mode='area'
            ).squeeze(1)
            
            pred = (pred * weighting_cuda).half()

            for tile_idx, (x0, x1, y0, y1) in enumerate(pos):
                global_pred[x0: x1, y0: y1] += pred[tile_idx]
                global_counter[x0: x1, y0: y1] += weighting

    for i in range(len(global_pred)):
        global_pred[i] = torch.div(global_pred[i], global_counter[i])

    return global_pred

# <h6> Step 3 - Set configuration and paths </h6>
BASE_PATH =  r'/N/slate/soodn/'
dataset = "colon" 
# dataset = "kidney" 
# dataset = "new-data"

DATA_PATH = BASE_PATH+'hubmap-'+dataset+'-segmentation'
IMG_PATH = DATA_PATH+'/test'


THRESHOLD = 0.5
USE_TTA = True # not DEBUG
OVERLAP_FACTOR = 1.5

CP_FOLDERS = [BASE_PATH+'/models/Kidney/b1_2cfix', BASE_PATH+'/models/Kidney/b1_last']
df = pd.read_csv(DATA_PATH+'/sample_submission.csv')
df_info = pd.read_csv(DATA_PATH+'/HuBMAP-20-dataset_information.csv')
rles = pd.read_csv(DATA_PATH+'/test.csv')

config = json.load(open(CP_FOLDERS[0]+'/config.json', 'r'))
config = Config(**config)
config.overlap_factor = OVERLAP_FACTOR


path_test = DATA_PATH+'/test'
models = []
for cp_folder in CP_FOLDERS:
    models += load_models(cp_folder)

DEBUG = False
for img in df['id'].unique():
    if DEBUG:  # Check performances on a validation image
        img = "2f6ecfcdf"  # check repro
#         img = "4ef6695ce" # biggest img
        IMG_PATH = DATA_PATH + "train"
        models = [models[0], models[5]]
                  
    
    print(f'\n\t Image {img}')
    
    print(f'\n - Building dataset')
    if dataset == "colon":
         rle_truth = rles[rles['id'] == img]["predicted"]
    elif dataset == "kidney":
        rle_truth = rles[rles['id'] == img]["encoding"]
 
    
    predict_dataset = InferenceEfficientDataset(
        f"{IMG_PATH}/{img}.tiff",
        rle=rle_truth,
        overlap_factor=config.overlap_factor,
        reduce_factor=config.reduce_factor,
        tile_size=config.tile_size,
        transforms=HE_preprocess(augment=False, visualize=False),
    )
    
    print(f'\n - Predicting masks')

    global_pred = predict_entire_mask(
        predict_dataset, models, batch_size=config.val_bs, tta=USE_TTA
    )
    
    del predict_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    print('\n - Encoding')
    
    global_pred_np = np.zeros(global_pred.size(), dtype=np.uint8)

    for i in range(global_pred_np.shape[0]):
        global_pred_np[i] = (global_pred[i] > THRESHOLD).cpu().numpy().astype(np.uint8)
    if dataset == "colon":
        shape = df_info[df_info.image_file == img][['width_pixels', 'height_pixels']].values.astype(int)[0]
    elif dataset == "kidney":
        shape = df_info[df_info.image_file == img+'.tiff'][['width_pixels', 'height_pixels']].values.astype(int)[0]
    rle = rle_encode_less_memory(global_pred_np)
    df.loc[df.id == img, 'predicted'] = rle
    
    if DEBUG:
        shape = df_info[df_info.image_file == img + ".tiff"][['width_pixels', 'height_pixels']].values.astype(int)[0]
        mask_truth = enc2mask(rle_truth, shape)
        score = dice_scores_img(global_pred_np, mask_truth)
        print(f" -> Scored {score :.4f} with threshold {THRESHOLD:.2f}")
        break
        
    del global_pred, global_pred_np
    torch.cuda.empty_cache()
    gc.collect()

df.to_csv(f'submission-{dataset}.csv', index=False)
print ("Run time = ", elapsed_time(start))