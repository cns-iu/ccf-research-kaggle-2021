
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/configer.ipynb

import os
from pathlib import Path

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUTS = ''
_C.INPUTS = ''

_C.DATA = CN()
_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.DATASETS = (0,)
_C.DATA.TRAIN.GPU_PRELOAD = False
_C.DATA.TRAIN.PRELOAD = False
_C.DATA.TRAIN.CACHE = False

_C.DATA.TRAIN.MULTIPLY = CN()
_C.DATA.TRAIN.MULTIPLY.rate = 0

_C.DATA.VALID = CN()
_C.DATA.VALID.DATASETS = (0,)
_C.DATA.VALID.PRELOAD = False
_C.DATA.VALID.GPU_PRELOAD = False
_C.DATA.VALID.CACHE = False
_C.DATA.VALID.MULTIPLY = 1

_C.DATA.TEST = CN()
_C.DATA.TEST.DATASETS = (0,)
_C.DATA.TEST.PRELOAD = False
_C.DATA.TEST.GPU_PRELOAD = False
_C.DATA.TEST.CACHE = False
_C.DATA.TEST.MULTIPLY = 1

_C.TRANSFORMERS = CN()
_C.TRANSFORMERS.TRAIN = CN()
_C.TRANSFORMERS.TRAIN.AUG=''

_C.TRANSFORMERS.VALID = CN()
_C.TRANSFORMERS.VALID.AUG=''

_C.TRANSFORMERS.TEST = CN()
_C.TRANSFORMERS.TEST.AUG=''

_C.TRANSFORMERS.CROP = -1,
_C.TRANSFORMERS.RESIZE = -1,


_C.MODEL = CN()


_C.TRAIN = CN()
_C.TRAIN.GPUS = (0,)
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.AMP_LEVEL = ''

_C.TRAIN.SAVE_STEP = 0
_C.TRAIN.SCALAR_STEP = 0
_C.TRAIN.TB_STEP = 0

_C.TRAIN.BATCH_SIZE = 0
_C.TRAIN.EPOCH_COUNT = 0
_C.TRAIN.EPOCH_DECAY_STEP = 0
_C.TRAIN.LR_DECAY = 0.0


_C.VALID = CN()
_C.VALID.STEP = 0
_C.VALID.NUM_WORKERS = 0
_C.VALID.BATCH_SIZE = 0

_C.TEST = CN()
_C.TEST.NUM_WORKERS = 0
_C.TEST.BATCH_SIZE = 0

_C.PARALLEL = CN()
_C.PARALLEL.DDP = False
_C.PARALLEL.LOCAL_RANK = 0
_C.PARALLEL.WORLD_SIZE = 1

cfg = _C

def cfg_init(path, aux_path='/mnt/configs', prefix='config_',freeze=True):
    cfg.merge_from_file(path)

    aux_path = Path(aux_path)
    if aux_path.exists():
        aux_cfgs = os.listdir(aux_path)

        if aux_cfgs:
            for aux_cfg in aux_cfgs:
                if aux_cfg.startswith('config_'):
                    p = aux_path/aux_cfg
                    cfg.merge_from_file(p)

    if freeze: cfg.freeze()

# def get_cfg_defaults():
#     """Get a yacs CfgNode object with default values for my_project."""
#     # Return a clone so that the defaults will not be altered
#     # This is for the "local variable" use pattern
#     return _C.clone()
