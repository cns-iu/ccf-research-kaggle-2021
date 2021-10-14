import os
from pathlib import Path

from yacs.config import CfgNode as CN

'''
yacs configurator : https://github.com/rbgirshick/yacs

This file is kinda template which joins with actual .yaml config (src/configs/unet.yaml)
In this file all values are intentionally WRONG, because one should init cfg by hand in code:

    from config import cfg, cfg_init
    cfg_init('unet.yaml')

, which will create a singleton object cfg to use throughout all process functions


All params is of specific types, i.e. you cant use .yaml like that:
    unet.yaml:
        TRAIN.BATCH_SIZE = 32.
    config.py:
        TRAIN.BATCH_SIZE = 16


Some of paramteres like WORLD_SIZE should not be inited by hand in .yaml file
Consider them technical information

'''


_C = CN()
_C.OUTPUTS = ''
_C.INPUTS = ''

_C.DATA = CN()

_C.DATA.TRAIN = CN() 
_C.DATA.TRAIN.DATASETS = (0,)
_C.DATA.TRAIN.FOLDS = (0,)
_C.DATA.TRAIN.GPU_PRELOAD = False
_C.DATA.TRAIN.PRELOAD = False
_C.DATA.TRAIN.CACHE = False
_C.DATA.TRAIN.MULTIPLY = CN()
_C.DATA.TRAIN.MULTIPLY.rate = 0

_C.DATA.VALID = CN()
_C.DATA.VALID.DATASETS = (0,)
_C.DATA.VALID.FOLDS = (0,)
_C.DATA.VALID.PRELOAD = False
_C.DATA.VALID.GPU_PRELOAD = False
_C.DATA.VALID.CACHE = False
_C.DATA.VALID.MULTIPLY = CN()
_C.DATA.VALID.MULTIPLY.rate = 0

_C.DATA.VALID2 = CN()
_C.DATA.VALID2.DATASETS = (0,)
_C.DATA.VALID2.PRELOAD = False
_C.DATA.VALID2.GPU_PRELOAD = False
_C.DATA.VALID2.CACHE = False
_C.DATA.VALID2.MULTIPLY = CN()
_C.DATA.VALID2.MULTIPLY.rate = 0

_C.DATA.SSL = CN()
_C.DATA.SSL.DATASETS = (0,)
_C.DATA.SSL.PRELOAD = False
_C.DATA.SSL.GPU_PRELOAD = False
_C.DATA.SSL.CACHE = False
_C.DATA.SSL.MULTIPLY = CN()
_C.DATA.SSL.MULTIPLY.rate = 0

_C.DATA.TEST = CN()
_C.DATA.TEST.DATASETS = (0,)
_C.DATA.TEST.PRELOAD = False
_C.DATA.TEST.GPU_PRELOAD = False
_C.DATA.TEST.CACHE = False
_C.DATA.TEST.MULTIPLY = 1

_C.TRANSFORMERS = CN()

_C.TRANSFORMERS.MEAN = (0,)
_C.TRANSFORMERS.STD = (0,)

_C.TRANSFORMERS.TRAIN = CN()
_C.TRANSFORMERS.TRAIN.AUG=''

_C.TRANSFORMERS.VALID = CN()
_C.TRANSFORMERS.VALID.AUG=''

_C.TRANSFORMERS.VALID2 = CN()
_C.TRANSFORMERS.VALID2.AUG=''

_C.TRANSFORMERS.SSL = CN()
_C.TRANSFORMERS.SSL.AUG=''

_C.TRANSFORMERS.TEST = CN()
_C.TRANSFORMERS.TEST.AUG=''

_C.TRANSFORMERS.CROP = (0,)
_C.TRANSFORMERS.CROP_VAL = (0,)
_C.TRANSFORMERS.RESIZE = (0,)


_C.MODEL = CN()

_C.TRAIN = CN()
_C.TRAIN.DOWNSCALE = 0
_C.TRAIN.NUM_FOLDS = 0
_C.TRAIN.FOLD_IDX = -1
_C.TRAIN.LRS = (0,)
_C.TRAIN.SELECTIVE_BP = 1.
_C.TRAIN.HARD_MULT = 0
_C.TRAIN.EMA = 0.
_C.TRAIN.WEIGHTS = ''
_C.TRAIN.INIT_MODEL = ''
_C.TRAIN.INIT_ENCODER = (0,)
_C.TRAIN.FREEZE_ENCODER = False
_C.TRAIN.AMP= False
_C.TRAIN.GPUS = (0,)
_C.TRAIN.NUM_WORKERS = 0

_C.TRAIN.SAVE_STEP = 0.
_C.TRAIN.SCALAR_STEP = 0
_C.TRAIN.TB_STEP = 0

_C.TRAIN.BATCH_SIZE = 0
_C.TRAIN.EPOCH_COUNT = 0

_C.VALID = CN()
_C.VALID.STEP = 0
_C.VALID.NUM_WORKERS = 0
_C.VALID.BATCH_SIZE = 0

_C.VALID2 = CN()
_C.VALID2.STEP = 0
_C.VALID2.NUM_WORKERS = 0
_C.VALID2.BATCH_SIZE = 0

_C.SSL = CN()
_C.SSL.STEP = 0
_C.SSL.NUM_WORKERS = 0
_C.SSL.BATCH_SIZE = 0

_C.TEST = CN()
_C.TEST.NUM_WORKERS = 0
_C.TEST.BATCH_SIZE = 0

_C.PARALLEL = CN()
_C.PARALLEL.DDP = False
_C.PARALLEL.LOCAL_RANK = -1
_C.PARALLEL.WORLD_SIZE = 0
_C.PARALLEL.IS_MASTER = False

cfg = _C
        
def cfg_init(path, freeze=True):
    cfg.merge_from_file(path)
    if freeze: cfg.freeze()
        
