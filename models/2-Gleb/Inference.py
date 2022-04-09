import gc
import torch
import random
import numpy as np
import pandas as pd

import cv2
import rasterio as rio

import logging
import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

from pathlib import Path
from functools import partial
from collections import OrderedDict

import psutil
from tqdm import tqdm
import albumentations as albu
import albumentations.pytorch as albu_pt

import segmentation_models_pytorch as smp
import ttach as tta

import time
start_time = time.time()

BASE_PATH = r'/N/slate/soodn/'
dataset = "colon" 
# dataset = "kidney" 
# dataset = "new-data"

def elapsed_time(start_time):
    return time.time() - start_time
    
class TFReader:
    """Reads tiff files.

    If subdatasets are available, then use them, otherwise just handle as usual.
    """

    def __init__(self, path_to_tiff_file: str):
        self.ds = rio.open(path_to_tiff_file)
        self.subdatasets = self.ds.subdatasets
        self.is_subsets_avail = len(self.subdatasets) > 1 # 0? WTF
        
        if self.is_subsets_avail:
            path_to_subdatasets = self.ds.subdatasets
            self.list_ds = [rio.open(path_to_subdataset)
                            for path_to_subdataset in path_to_subdatasets]

    def read(self, window=None, boundless=True):
        if window is not None: ds_kwargs = {'window':window, 'boundless':boundless}
        else: ds_kwargs = {}
        if self.is_subsets_avail:
            t = [ds.read(**ds_kwargs) for ds in self.list_ds]
            output = np.vstack(t)
        else:
            output = self.ds.read(**ds_kwargs)
        return output

    @property
    def shape(self):
        return self.ds.shape

    def __del__(self):
        del self.ds
        if self.is_subsets_avail:
            del self.list_ds
    
    def close(self):
        self.ds.close()
        if self.is_subsets_avail:
            for i in range(len(self.list_ds)):
                self.list_ds[i].close()

def get_images(train=False):
    p = 'train' if train else 'test'
    return list(Path(fr'{BASE_PATH}hubmap-{dataset}-segmentation/{p}').glob('*.tiff'))

def get_random_crops(n=8, ss=512):
    imgs = get_images(False)
    img = random.choice(imgs)
    tfr = TFReader(img)
    W,H = tfr.shape
    for i in range(n):
        x,y,w,h = random.randint(5000, W-5000),random.randint(5000, H-5000), ss,ss
        res = tfr.read(window=((y,y+h),(x,x+w)))
        yield res
    tfr.close()
        
def test_tf_reader():
    window=((5000,5100),(5000,5100))
    imgs = get_images(False)
    for img in imgs:
        tfr = TFReader(img)
        res = tfr.read(window=window)
        tfr.close()

def mb_to_gb(size_in_mb): return round(size_in_mb / (1024 * 1024 * 1024), 3)

def get_ram_mems(): 
    total, avail, used = mb_to_gb(psutil.virtual_memory().total), \
                         mb_to_gb(psutil.virtual_memory().available), \
                         mb_to_gb(psutil.virtual_memory().used)
    return f'Total RAM : {total} GB, Available RAM: {avail} GB, Used RAM: {used} GB'

def load_model(model, model_folder_path):
    model = _load_model_state(model, model_folder_path)
    model.eval()
    return model

def _load_model_state(model, path):
    path = Path(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))['model_state']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            k = k.lstrip('module')[1:]
        new_state_dict[k] = v
    del state_dict
    model.load_state_dict(new_state_dict)
    del new_state_dict
    return model

class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params): return torch.from_numpy(mask).permute((2,0,1))
    def apply(self, image, **params): return torch.from_numpy(image).permute((2,0,1))

def rescale(batch_img, scale):
    return torch.nn.functional.interpolate(batch_img, scale_factor=(scale, scale), mode='bilinear', align_corners=False)
    
class MegaModel():
    def __init__(self, model_folders, use_tta=True, use_cuda=True, threshold=.5, half_mode=False):
        self.use_cuda = use_cuda
        self.threshold = threshold
        self.use_tta = use_tta
        self.half_mode = half_mode
        self.averaging = 'mean'
        
        self._model_folders = model_folders#list(Path(root).glob('*')) # root / model1 ; model2; model3; ...
        # TODO : SORT THEM
#         self._model_types = [
#                 smp.Unet(encoder_name='timm-regnety_016', encoder_weights=None),
#                 smp.Unet(encoder_name='timm-regnety_016', encoder_weights=None, decoder_attention_type='scse')
#                 ]
        self._model_types = [
                smp.UnetPlusPlus(encoder_name='timm-regnety_016', encoder_weights=None, decoder_attention_type='scse'),
                smp.Unet(encoder_name='timm-regnetx_032', encoder_weights=None),    
                smp.Unet(encoder_name='timm-regnety_016', encoder_weights=None),
                smp.Unet(encoder_name='timm-regnety_016', encoder_weights=None, decoder_attention_type='scse')
                ]
        self._model_scales = [3,3,3,3]
        assert len(self._model_types) == len(self._model_folders)
        assert len(self._model_scales) == len(self._model_folders)
        
        self.models = self.models_init()
        mean, std = [0.6226 , 0.4284 , 0.6705], [ 0.1246 , 0.1719 , 0.0956]
        self.itransform = albu.Compose([albu.Normalize(mean=mean, std=std), ToTensor()])
        
    def models_init(self):
        models = {}
        tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90])
        
        ]
        )
        for mt, mf, ms in zip(self._model_types, self._model_folders, self._model_scales):
            mf = Path(mf)
            model_names = list(mf.rglob('*.pth'))
            mm = []
            for mn in model_names:
                m = load_model(mt, mn)
                if self.use_tta : m = tta.SegmentationTTAWrapper(m, tta_transforms, merge_mode='mean')
                if self.use_cuda: m = m.cuda()
                if self.half_mode: m = m.half()
                mm.append(m)
            models[mf] = {'scale': ms, 'models':mm}
        return models

    def _prepr(self, img, scale):
        ch, H, W, dtype = *img.shape, img.dtype
        assert ch==3 and dtype==np.uint8
        img = img.transpose(1,2,0)
        img = cv2.resize(img, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
        return self.itransform(image=img)['image']

    def each_forward(self, model, scale, batch):
        with torch.no_grad():
            res = model(batch)
            res = torch.sigmoid(res)
        res = rescale(res, scale)
        return res

    def __call__(self, imgs, cuda=True):
        batch = None
        scale = None
        preds = []

        for mod_name, params in self.models.items(): # [{'a/b/c/1231.pth':{'type':Unet,'scale':3 }}, .. ]
            _scale = params['scale']
            if batch is None or scale != _scale:
                scale = _scale
                batch = [self._prepr(i, scale) for i in imgs]
                batch = torch.stack(batch, axis=0)
                if self.half_mode: batch = batch.half()
                if self.use_cuda: batch = batch.cuda()
            models = params['models']
            _predicts = torch.stack([self.each_forward(m, scale, batch) for m in models])
            preds.append(_predicts)

        res = torch.vstack(preds).mean(0) # TODO : mean(0)? do right thing 
        res = res > self.threshold
        return  res

class Dummymodel(torch.nn.Module):
    def forward(self, x):
        x[x<.5]=0.9
        return x
    
def infer(imgs, model):
    assert isinstance(imgs, list)
    with torch.no_grad():
        predicts = model(imgs)
    return predicts
    
def get_infer_func(model_folders, use_tta, use_cuda, threshold):
    m = MegaModel(model_folders=model_folders, use_tta=use_tta, use_cuda=use_cuda, threshold=threshold)
    return partial(infer, model=m)

model_index = 8
model_folders = [f'{BASE_PATH}models/models{model_index}/UnetPlusPlus_timm-regnety_016_scse',
                 f'{BASE_PATH}models/models{model_index}/Unet_timm-regnetx_032',
                 f'{BASE_PATH}models/models{model_index}/Unet_timm-regnety_016',
                 f'{BASE_PATH}models/models{model_index}/Unet_timm-regnety_016_scse'
                ]

def calc_infer_appr_time(trials, block_size, batch_size, model_folders, tta=True, use_cuda=True, scale = 3):
    block_size = block_size * 3
    pad = block_size // 4
    f = get_infer_func(model_folders, use_tta=tta, use_cuda=use_cuda, threshold=.5)
    for trial in tqdm(range(trials), position=0, leave=True): 
        imgs = list(get_random_crops(batch_size, block_size + 2*pad))
        res = f(imgs).cpu()

def get_gpu_mems():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    total, avail, used = mb_to_gb(info.total), \
                         mb_to_gb(info.free), \
                         mb_to_gb(info.used)
    nvidia_smi.nvmlShutdown()
    return f'Total GPU mem: {total} GB, Available GPU mem: {avail} GB, Used GPU mem: {used} GB'

def _count_blocks(dims, block_size):
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks

def generate_block_coords(H, W, block_size):
    h,w = block_size
    nYBlocks = (int)((H + h - 1) / h)
    nXBlocks = (int)((W + w - 1) / w)
    
    for X in range(nXBlocks):
        cx = X * h
        for Y in range(nYBlocks):
            cy = Y * w
            yield cy, cx, h, w

def pad_block(y,x,h,w, pad): return np.array([y-pad, x-pad, h+2*pad, w+2*pad])
def crop(src, y,x,h,w): return src[..., y:y+h, x:x+w]
def crop_rio(ds, y,x,h,w):
    block = ds.read(window=((y,y+h),(x,x+w)), boundless=True)
    #block = np.zeros((3, h, w), dtype = np.uint8)
    return block

def paste(src, block, y,x,h,w):src[..., y:y+h, x:x+w] = block
def paste_crop(src, part, block_cd, pad):
    _,H,W = src.shape
    y,x,h,w = block_cd
    h, w = min(h, H-y), min(w, W-x)  
    part = crop(part, pad, pad, h, w)
    paste(src, part, *block_cd)
         
def mp_func_wrapper(func, args): return func(*args)
def chunkify(l, n): return [l[i:i + n] for i in range(0, len(l), n)]

def mask2rle(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def infer_blocks(blocks, do_inference):
    blocks = do_inference(blocks).cpu().numpy()    
    if isinstance(blocks, tuple): blocks = blocks[0]
    return blocks#.astype(np.uint8)

def start(img_name, do_inference):
    logger.info('Start')
    
    scale = 3
    s = 512
    block_size = s * scale
    batch_size = 6
    pad = s//4 * scale
    
    ds = TFReader(str(img_name))
    H, W = ds.shape
    cds = list(generate_block_coords(H, W, block_size=(block_size, block_size)))
    #cds = cds[:36]
    total_blocks = len(cds)
    
    mask = np.zeros((1,H,W)).astype(np.bool)
    count = 0
    batch = []
    
    for block_cd in tqdm(cds):
        if len(batch) == batch_size:
            blocks = [b[0] for b in batch]
            block_cds = [b[1] for b in batch]
            # logger.info(get_gpu_mems())
            block_masks = infer_blocks(blocks, do_inference)
            [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, _ in zip(block_masks, block_cds, blocks)]
            batch = []
            gc.collect()
        
        padded_block_cd = pad_block(*block_cd, pad)
        block = crop_rio(ds, *(padded_block_cd))
        batch.append((block, block_cd))
        count+=1
            
    if batch:
        blocks = [b[0] for b in batch]
        block_cds = [b[1] for b in batch]
        # logger.info(get_gpu_mems())
        block_masks = infer_blocks(blocks, do_inference)
        [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, _ in zip(block_masks, block_cds, blocks)]
        batch = []
    
    ds.close()
    rle = mask2rle(mask)
    return rle, mask

# %%
from py3nvml.py3nvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)

logger = logging.getLogger('dev')
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler('log.log')
fileHandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

model_folders = [f'{BASE_PATH}models/models{model_index}/UnetPlusPlus_timm-regnety_016_scse',
                 f'{BASE_PATH}models/models{model_index}/Unet_timm-regnetx_032',
                 f'{BASE_PATH}models/models{model_index}/Unet_timm-regnety_016',
                 f'{BASE_PATH}models/models{model_index}/Unet_timm-regnety_016_scse'
                ]

threshold = 0.55
use_tta=True
use_cuda=True
do_inference = get_infer_func(model_folders, use_tta=use_tta, use_cuda=use_cuda, threshold=threshold)

imgs = get_images()
subm = {}

idx = 0
for img_name in imgs[:]:
    rle, mask = start(img_name, do_inference=do_inference)    
    subm[img_name.stem] = {'id':img_name.stem, 'predicted': rle}
    idx+=1

df_sub = pd.DataFrame(subm).T
df_sub.to_csv(f'submission-{dataset}.csv', index=False)
print ("Run time = ", elapsed_time(start))