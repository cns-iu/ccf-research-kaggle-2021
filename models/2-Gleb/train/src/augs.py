from functools import partial

import torch  # to tensor transform
import albumentations as albu
import albumentations.pytorch as albu_pt

from _add_augs import AddLightning, AddFakeGlom, AddGammaCorrection


class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params): return torch.from_numpy(mask).permute((2,0,1))
    def apply(self, image, **params): return torch.from_numpy(image).permute((2,0,1))

class Augmentator:
    def __init__(self, cfg, compose):
        self.cfg = cfg 
        self.resize_w, self.resize_h = self.cfg.RESIZE
        self.crop_w, self.crop_h = self.cfg.CROP
        self.crop_val_w, self.crop_val_h = self.cfg.CROP_VAL if self.cfg.CROP_VAL is not (0,) else seld.cfg.CROP
        self.compose = compose
        
        self.mean = self.cfg.MEAN if self.cfg.MEAN is not (0,) else (0.46454108, 0.43718538, 0.39618185)
        self.std = self.cfg.STD if self.cfg.STD is not (0,) else (0.23577851, 0.23005974, 0.23109385)
    

    def get_aug(self, kind):
        if kind == 'val': return self.aug_val()
        elif kind == 'val_forced': return self.aug_val_forced()
        elif kind == 'test': return self.aug_test ()
        elif kind == 'light': return self.aug_light()
        elif kind == 'light_scale': return self.aug_light_scale()
        elif kind == 'blank': return self.aug_blank()
        elif kind == 'resize': return self.resize()
        elif kind == 'wocrop': return self.aug_wocrop()
        elif kind == 'ssl': return self.aug_ssl()
        else: raise Exception(f'Unknown aug : {kind}')
        
    def norm(self): return self.compose([albu.Normalize(mean=self.mean, std=self.std), ToTensor()])
    def rand_crop(self): albu.RandomCrop(self.crop_h,self.crop_w)
    def crop_scale(self): return albu.RandomResizedCrop(self.crop_h, self.crop_w, scale=(.3,.7))
    def resize(self): return albu.Resize(self.resize_h, self.resize_w)
    def blur(self, p): return albu.OneOf([
                        albu.GaussianBlur((3,9)),
                        #albu.MotionBlur(),
                        #albu.MedianBlur()
                    ], p=p)
    def scale(self, p): return albu.ShiftScaleRotate(0.1, 0.2, 45, p=p)
    def cutout(self, p): return albu.OneOf([
            albu.Cutout(24, 64, 64, 0, p=.3),
            albu.GridDropout(0.5, fill_value=230, random_offset=True, p=.3),
            albu.CoarseDropout(24, 64, 64, 16, 16, 16, 220, p=.4)
        ],p=p)
    
    def d4(self): return self.compose([
                    albu.Flip(),
                    albu.RandomRotate90()
                    ]) 

    def multi_crop(self): return albu.OneOf([
                    albu.CenterCrop(self.crop_h, self.crop_w, p=.2),
                    albu.RandomCrop(self.crop_h, self.crop_w, p=.8),
                    ], p=1)    

    def color_jit(self, p): return albu.OneOf([
                    albu.HueSaturationValue(10,15,10),
                    albu.CLAHE(clip_limit=4),
                    albu.RandomBrightnessContrast(.4, .4),
                    albu.ChannelShuffle(),
                    ], p=p)

    def aug_ssl(self): return self.compose([
                                self._ssl(p=1.),
                                self.norm()
                                ])
    
    def _ssl(self, p): return self.compose([
                    self.FakeGlo(p=.3),
                    self.Alights(p=.7),
                    albu.OneOf([
                        albu.HueSaturationValue(30,40,30),
                        albu.CLAHE(clip_limit=4),
                        albu.RandomBrightnessContrast((-0.5, .5), .3),
                        albu.ColorJitter(brightness=.5, contrast=0.5, saturation=0.3, hue=0.3)
                    ], p=.5),
                    self.cutout(p=.3),
                    self.blur(p=.2),
                    ], p=p)

    def aug_light_scale(self): return self.compose([
                                                    self.multi_crop(), 
                                                    self.d4(),
                                                    self.additional_res(),
                                                    self.norm()
                                                    ])

    def custom_augs(self, p): return albu.OneOf([
                    AddGammaCorrection(p=.3),
        ], p=p)

    def additional_res(self):
        return self.compose([
                    self.scale(p=.2),
                    self.custom_augs(p=.1),
                    self.color_jit(p=.3),
                    self.cutout(p=.1),
                    self.blur(p=.2),
            ], p=.4)

    def aug_val_forced(self): return self.compose([albu.CropNonEmptyMaskIfExists(self.crop_h,self.crop_w), self.norm()])
    def aug_val(self): return self.compose([albu.CenterCrop(self.crop_val_h,self.crop_val_w), self.norm()])
    def aug_light(self): return self.compose([albu.CenterCrop(self.crop_h,self.crop_w, p=1), albu.Flip(), albu.RandomRotate90(), self.norm()])
    def aug_wocrop(self): return self.compose([self.resize(), albu.Flip(), albu.RandomRotate90(), self.norm()])
    def aug_blank(self): return self.compose([self.resize()])
    def aug_test(self): return self.compose([self.resize(), self.norm()])


def get_aug(aug_type, transforms_cfg, tag=False):
    """ aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
        transforms_cfg (dict): part of main cfg
    """
    compose = albu.Compose #if not tag else partial(albu.Compose, additional_targets={'mask1':'mask', 'mask2':'mask'})
    auger = Augmentator(cfg=transforms_cfg, compose=compose)
    return auger.get_aug(aug_type)

