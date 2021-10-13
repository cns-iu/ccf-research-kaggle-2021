# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,shallow//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
from functools import partial
import albumentations as albu
import albumentations.pytorch as albu_pt

import torch
import numpy as np

# %% tags=["active-ipynb"]
# import matplotlib.pyplot as plt

# %% tags=["active-ipynb"]
# %load_ext autoreload
# %autoreload 2
#
# %matplotlib inline

# %% [markdown]
# # Code

# %%
BBOX_PARAMS = {
    "format":'coco',
    "label_fields":None,
    "min_area":0.0,
    "min_visibility":0.0,
}
def composer(using_boxes): return albu.Compose if not using_boxes else partial(albu.Compose, bbox_params=albu.BboxParams(**BBOX_PARAMS)) 

class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask).permute((2,0,1))
    
    def apply(self, image, **params):
        return torch.from_numpy(image).permute((2,0,1))

class Augmentator:
    def __init__(self, cfg, using_boxes=False):
        self.cfg = cfg 
        self.using_boxes = using_boxes
        self.resize_w, self.resize_h = self.cfg['RESIZE']
        self.crop_w, self.crop_h = self.cfg['CROP']
        self.compose = composer(using_boxes)

        self.mean = self.cfg.MEAN if self.cfg.MEAN is not (0,) else (0.46454108, 0.43718538, 0.39618185)
        self.std = self.cfg.STD if self.cfg.STD is not (0,) else (0.23577851, 0.23005974, 0.23109385)
        
    
    def get_aug(self, kind):
        if kind == 'val': return self.aug_val()
        elif kind == 'val_forced': return self.aug_val_forced()
        elif kind == 'test': return self.aug_test ()
        elif kind == 'light': return self.aug_light()
        else: raise Exception(f'Unknown aug : {kind}')
        
    def norm(self): 
        return self.compose([albu.Normalize(mean=self.mean, std=self.std), ToTensor()])
    
    def rand_crop(self):
        return albu.OneOf([
                #albu.RandomResizedCrop(h,w, scale=(0.05, 0.4)), 
                albu.RandomCrop(self.crop_h,self.crop_w)
                #albu.CropNonEmptyMaskIfExists(h,w)
            ], p=1)    
    
    def resize(self): return albu.Resize(self.resize_h, self.resize_w)
    
    def aug_val(self): return self.compose([albu.CenterCrop(self.crop_h,self.crop_w), self.norm()])
    def aug_val_forced(self): return self.compose([albu.CropNonEmptyMaskIfExists(self.crop_h,self.crop_w), self.norm()])
    
    def aug_test(self): return self.compose([albu.Resize(self.resize_h,self.resize_w), self.norm()])
    def aug_light(self): return self.compose([self.rand_crop(), albu.Flip(), albu.RandomRotate90(), self.norm()])
    
    def aug_medium(self): 
        return albu.Compose([
                            self.rand_crop(),
                            albu.Flip(),
                            albu.ShiftScaleRotate(),  # border_mode=cv2.BORDER_CONSTANT
                            # Add occasion blur/sharpening
                            albu.OneOf([albu.GaussianBlur(), albu.IAASharpen(), albu.NoOp()]),
                            # Spatial-preserving augmentations:
                            # albu.OneOf([albu.CoarseDropout(), albu.MaskDropout(max_objects=5), albu.NoOp()]),
                            albu.GaussNoise(),
                            albu.OneOf(
                                [
                                    albu.RandomBrightnessContrast(),
                                    albu.CLAHE(),
                                    albu.HueSaturationValue(),
                                    albu.RGBShift(),
                                    albu.RandomGamma(),
                                ]
                            ),
                            # Weather effects
                            albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
                            self.norm(),
                        ])
    def aug_hard(self):
        return albu.Compose([  
                            self.rand_crop(), 
                            albu.RandomRotate90(),
                            albu.Transpose(),
                            albu.RandomGridShuffle(p=0.2),
                            albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.2),
                            albu.ElasticTransform(alpha_affine=5, p=0.2),
                            # Add occasion blur
                            albu.OneOf([albu.GaussianBlur(), albu.GaussNoise(), albu.IAAAdditiveGaussianNoise(), albu.NoOp()]),
                            # D4 Augmentations
                            albu.OneOf([albu.CoarseDropout(), albu.NoOp()]),
                            # Spatial-preserving augmentations:
                            albu.OneOf(
                                [
                                    albu.RandomBrightnessContrast(brightness_by_max=True),
                                    albu.CLAHE(),
                                    albu.HueSaturationValue(),
                                    albu.RGBShift(),
                                    albu.RandomGamma(),
                                    albu.NoOp(),
                                ]
                            ),
                            # Weather effects
                            albu.OneOf([albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), albu.NoOp()]),
                            self.norm(),
                        ])


# %%
def get_aug(aug_type, transforms_cfg, using_boxes):
    """ aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
        transforms_cfg (dict): part of main cfg
    """
    auger = Augmentator(cfg=transforms_cfg, using_boxes=using_boxes)
    return auger.get_aug(aug_type)

# %%

# %%

# %%

# %%

# %%
