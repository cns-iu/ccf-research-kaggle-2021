import sys
import yaml
from PIL import Image
from pathlib import Path
from functools import partial

import cv2
import torch
import numpy as np
import ttach as tta
import albumentations as albu
import albumentations.pytorch as albu_pt

import augs

class CfgParse:
    def __init__(self, cfg_file):
        self.compose = albu.Compose
        data = self.parse_yaml(cfg_file)
        self.mean = data['TRANSFORMERS']['MEAN']
        self.std = data['TRANSFORMERS']['STD']
        self.num_folds = data['TRAIN']['NUM_FOLDS']
        self.scale = self.get_scale(data)
        assert self.scale is not None

    def get_scale(self, data):
        return  data['TRAIN'].get('DOWNSCALE', None)

    def parse_yaml(self, cfg_file):
        with open(cfg_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return data

    def norm(self): 
        return self.compose([albu.Normalize(mean=self.mean, std=self.std), augs.ToTensor()])

def parse_model_path(p):
    name = str(p.name)
    epoch = name.split('_')[0]
    return int(epoch[1:])

def get_last_model(src):
    # assumes that model name is of type e500_blabla.pth, sorted by epoch #500
    models = list(Path(src).glob('*.pth'))
    res = []
    for i, m in enumerate(models):
        epoch = parse_model_path(m)
        res.append([i,epoch])
    idx = sorted(res, key=lambda x: -x[1])[0][0]
    return models[idx]

def rescale(batch_img, scale): 
    return torch.nn.functional.interpolate(batch_img, scale_factor=(scale, scale), mode='bilinear', align_corners=False)

def preprocess_image(img, scale, transform):
    ch, H,W, dtype = *img.shape, img.dtype
    assert ch==3 and dtype==np.uint8
    img = img.transpose(1,2,0)
    img = cv2.resize(img, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    return transform(image=img)['image']

def _infer_func(imgs, transform, scale, model):
    batch = [preprocess_image(i, scale, transform) for i in imgs]
    batch = torch.stack(batch,axis=0).cuda()
    #print(batch.shape, batch.dtype, batch.max(), batch.min())
    with torch.no_grad():
        res = model(batch)
        if isinstance(res, tuple): res = res[0]
        res = torch.sigmoid(res)

    #print(res.shape, res.max(), res.min())
    res = rescale(res, scale)
    return res
    
def get_model(root, cfg_data):
    import importlib.util
    spec = importlib.util.spec_from_file_location("model", str(root/'src/model.py'))
    nn_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nn_model)

    num_folds = cfg_data.num_folds
    if num_folds <=1:
        model_path = get_last_model(root / 'models')
        print(f'Single Fold: Selecting model: {model_path}')
        model = nn_model.load_model('', str(model_path))
    else:
        models = []
        for i in range(num_folds):
            model_path = get_last_model(root / f'fold_{i}' /'models')
            print(f'Fold #{i}; Selecting model: {model_path}')
            model = nn_model.load_model(None, str(model_path))
            models.append(model.cuda())
        model = nn_model.FoldModel(models)
    model = model.cuda()
    return model

def get_infer_func(root, use_tta=False):
    """
        Wraps model and preprocessing in one function with argument [img, ...]
        img is HWC (because of transforms)
        Returns torch tensor on CUDA, [len(inp), 1, H, W]
    """
    root = Path(root)
    cfg_data = CfgParse(root/'cfg.yaml')
    transform = cfg_data.norm() 
    model = get_model(root, cfg_data)
    aux_cl = True
    if aux_cl: model = SegModel(model)
    if use_tta: 
        #transforms = tta.Compose( [
        #        tta.HorizontalFlip(),
        #        #tta.VerticalFlip(),
        #        tta.Rotate90(angles=[0, 90]),
        #        #tta.Scale(scales=[1, .5]),
        #        tta.Multiply(factors=[0.9, 1, 1.1]),
        #    ])
        transforms = tta.aliases.d4_transform()
        model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')

    return partial(_infer_func, transform=transform, scale=cfg_data.scale, model=model)


class SegModel(torch.nn.Module):
    def __init__(self, model):
        super(SegModel, self).__init__()
        self.m = model
        
    def forward(self, x):
        res = self.m(x)
        if isinstance(res, tuple): res = res[0]
        return res


def load_model(model, model_folder_path):
    model = _load_model_state(model, model_folder_path)
    model.eval()
    return model

def _load_model_state(model, path):
    path = Path(path)
    state_dict = torch.load(path)['model_state']
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

class MegaModel(torch.nn.Module):
    def __init__(self, root):
        self.averaging = 'mean'
        self._model_folders = list(Path(root).glob()) # root / model1 ; model2; model3; ...
        # TODO : SORT THEM
        self._model_types = [
                smp.Unet(encoder_name='timm-regnety_016'), # model1
                smp.Unet(encoder_name='timm-regnetx_032'), # model2
                smp.Unet(encoder_name='timm-regnetx_016'),
                smp.UnetPlusPlus(encoder_name='timm-regnety_016'),
                ]
        self._model_scales = [2,2,2,2]
        assert len(self._model_types) == len(self._model_folders)
        self.models = self.models_init()
        mean, std = [0.6226 , 0.4284 , 0.6705], [ 0.1246 , 0.1719 , 0.0956]
        self.itransform = albu.compose([albu.Normalize(mean=mean, std=std), ToTensor()])

    def models_init(self):
        models = {}
        tta_transforms = tta.aliases.d4_transform()

        for mt, mf, ms in zip(self._model_types, self._model_folders, self._model_scales):
            model_names = mf.glob('*.pth')
            mm = []
            for mn in model_names:
                m = load_model(mt, mn)
                m = tta.SegmentationTTAWrapper(m, tta_transforms, merge_mode='mean')
                mm.append(m)
            models[mf] = {'scale': ms, 'models':mm}
        return models

    def _prepr(self, img, scale):
        ch, H,W, dtype = *img.shape, img.dtype
        assert ch==3 and dtype==np.uint8
        img = img.transpose(1,2,0)
        img = cv2.resize(img, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
        return self.itransform(image=img)['image']

    def each_forward(self, model, scale, batch):
        #print(batch.shape, batch.dtype, batch.max(), batch.min())
        with torch.no_grad():
            res = model(batch)
            res = torch.sigmoid(res)
        res = rescale(res, scale)
        return res

    def forward(self, imgs):
        batch = None
        scale = None
        preds = []

        for mod_dict in self.models: # [{'a/b/c/1231.pth':{'type':Unet,'scale':3 }}, .. ]
            _scale = mod_dict['scale']
            if batch is None or scale != _scale:
                scale = _scale
                batch = [self._prepr(i, scale) for i in imgs]
                batch = torch.stack(batch, axis=0).cuda()

            models = mod_dict['models']
            _predicts = torch.stack([self.each_forward(m, scale, batch) for m in models])
            preds.append(_predicts)

        res = torch.stack(preds).mean(0) # TODO : mean(0)? do right thing 
        return  res


