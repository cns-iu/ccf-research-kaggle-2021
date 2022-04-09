from pathlib import Path
from functools import partial
from collections import OrderedDict
from logger import logger

import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import segmentation_models_pytorch as smp

from add_model import *

def to_cuda(models): return [model.cuda() for model in models]
def scale_lr(lr, cfg): return lr * float(cfg.TRAIN.BATCH_SIZE * cfg.PARALLEL.WORLD_SIZE)/256.
def sync_bn(models): return [apex.parallel.convert_syncbn_model(m) for m in models]
def get_trainable_parameters(model): return model.parameters()
def get_lr(**kwargs): return kwargs.get('lr', 1e-4)
def dd_parallel(models): return [DistributedDataParallel(m) for m in models]


def model_select(cfg):

    # default model
    model_list = [
        partial(smp.Unet, encoder_name='timm-regnety_016'),
        partial(smp.Unet, encoder_name='timm-regnetx_032'),
        partial(smp.Unet, encoder_name='timm-regnety_016', decoder_attention_type='scse'),
        partial(smp.UnetPlusPlus, encoder_name='timm-regnety_016', decoder_attention_type='scse')
        ]
    model_name_list = ['Unet_timm-regnety_016',
                       'Unet_timm-regnetx_032',
                       'Unet_timm-regnety_016_scse',
                       'UnetPlusPlus_timm-regnety_016_scse']

    model_index = 0
    model = model_list[model_index]
    print('Default model built, type: ', model_name_list[model_index])

    # select type from cfg.TRAIN.INIT_MODEL
    if cfg.TRAIN.INIT_MODEL:
        model_name = cfg.TRAIN.INIT_MODEL.split('/')[-1]
        print('Loading pre-trained model built, type: ', model_name)
        if model_name in model_name_list:
            model = model_list[model_name_list.index(model_name)]
        else:
            print('Loading failed: ', model_name)
            
    #model = partial(smp.Unet, encoder_name='timm-regnety_016', decoder_attention_type='scse')
    #model = partial(smp.Unet, encoder_name='timm-regnetx_032')
    #model = partial(smp.UnetPlusPlus, encoder_name='timm-regnety_016', decoder_attention_type='scse')
    #model = partial(smp.Unet, encoder_name='timm-regnety_016')
    #model = partial(smp.UnetPlusPlus, encoder_name='timm-regnety_016')
    #model = partial(smp.Unet, encoder_name='se_resnet50')
    return model

def build_model(cfg):
    model = model_select(cfg)()
    if cfg.TRAIN.INIT_MODEL: 
        logger.log('DEBUG', f'Init model: {cfg.TRAIN.INIT_MODEL}') 
        #model = _load_model_state(model, cfg.TRAIN.INIT_MODEL)
    elif cfg.TRAIN.INIT_ENCODER != (0,): 
        if cfg.TRAIN.FOLD_IDX == -1: enc_weights_name = cfg.TRAIN.INIT_ENCODER[0]
        else: enc_weights_name = cfg.TRAIN.INIT_ENCODER[cfg.TRAIN.FOLD_IDX]
        _init_encoder(model, enc_weights_name)
    else: pass

    model = model.cuda()
    model.train()
    return model 

def _init_encoder(model, src):
    logger.log('DEBUG', f'Init encoder: {src}') 
    enc_state = torch.load(src)['model_state']
    if "head.fc.weight" not in enc_state: 
        enc_state["head.fc.weight"] = None
        enc_state["head.fc.bias"] = None
    model.encoder.load_state_dict(enc_state)

def get_optim(cfg, model):
    base_lr = 1e-4# should be overriden in LR scheduler anyway
    lr = base_lr if not cfg.PARALLEL.DDP else scale_lr(base_lr, cfg) 
    
    opt = optim.AdamW
    opt_kwargs = {'amsgrad':True, 'weight_decay':1e-3}
    optimizer = opt(tencent_trick(model), lr=lr, **opt_kwargs)
    # if cfg.TRAIN.INIT_MODEL: 
    #     st =  _load_opt_state(model, cfg.TRAIN.INIT_MODEL)
    #     optimizer.load_state_dict(st)
    return optimizer

def wrap_ddp(cfg, model):
    if cfg.PARALLEL.DDP: 
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, 
                                    device_ids=[cfg.PARALLEL.LOCAL_RANK],
                                    find_unused_parameters=True,
                                    broadcast_buffers=True)
    return model


def load_model(cfg, model_folder_path, eval_mode=True):
    print(model_folder_path)
    # model_select syncing build and load, probably should be in cfg, by key as in datasets
    model = model_select(cfg)()
    model = _load_model_state(model, model_folder_path)
    if eval_mode: model.eval()
    return model

def _load_opt_state(model, path):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    opt_state = torch.load(path)['opt_state']
    return opt_state

def _load_model_state(model, path):
    path = Path(path)
    if path.suffix != '.pth': path = get_last_model_name(path)
    print(path)
    state_dict = torch.load(path)['model_state']

    # Strip ddp model TODO dont save it like that
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            k = k.lstrip('module')[1:]
        new_state_dict[k] = v
    del state_dict
    
    model.load_state_dict(new_state_dict)
    del new_state_dict
    print("Loaded: ", path)
    return model

