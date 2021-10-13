import os
import gc
import time
from pathlib import Path
from functools import partial

import torch
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
import segmentation_models_pytorch as smp

import data
import loss
import utils
import shallow as sh

from logger import logger
from config import cfg, cfg_init
from model import build_model, wrap_ddp, get_optim
from callbacks import *


def clo(logits, targets, crit):
    l1 = crit(logits, targets)
    l2 = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return l2 + l1

def sbce(logits, targets, reduction='none'):
    cr = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=.1)
    return cr(logits, targets)

def start(cfg, output_folder):
    datasets = data.build_datasets(cfg, dataset_types=['TRAIN', 'VALID'])
    n = cfg.TRAIN.NUM_FOLDS
    if n <= 1: start_fold(cfg, output_folder, datasets)
    else: 
        datasets_folds = data.make_datasets_folds(cfg, datasets, n)
        for i, i_datasets in enumerate(datasets_folds):
            cfg["TRAIN"]["FOLD_IDX"] = i
            if cfg.PARALLEL.IS_MASTER: 
                print(f'\n\nFOLD # {i}\n\n')
                utils.dump_params(cfg, output_folder)

            fold_output = None
            if output_folder:
                fold_output = output_folder / f'fold_{i}'
                fold_output.mkdir()
            start_fold(cfg, fold_output, i_datasets) 
            if cfg.PARALLEL.IS_MASTER: print(f'\n\n END OF FOLD # {i}\n\n')


def start_fold(cfg, output_folder, datasets):
    n_epochs = cfg.TRAIN.EPOCH_COUNT
    selective = False
    dls = data.build_dataloaders(cfg, datasets, selective=selective)
    
    print("DLS: ", len(dls), dls)
    model = build_model(cfg)
    model_ema = ema.ModelEmaV2(model, decay=cfg.TRAIN.EMA)
    model = wrap_ddp(cfg, model)
    opt = get_optim(cfg, model)
    if cfg.TRAIN.FREEZE_ENCODER: unwrap_model(model).encoder.requires_grad_(False)

    
    criterion = partial(clo, crit=smp.losses.DiceLoss('binary'))

    train_cb = TrainCB(logger=logger) 
    val_cb = ValEMACB(model_ema=model_ema, logger=logger) if cfg.TRAIN.EMA else ValCB(logger=logger)
    
    if cfg.PARALLEL.IS_MASTER:
        utils.dump_params(cfg, output_folder)
        models_dir = output_folder / 'models'
        tb_dir = output_folder / 'tb'
        models_dir.mkdir()
        tb_dir.mkdir()

        step = cfg.TRAIN.TB_STEP
        writer = SummaryWriter(log_dir=tb_dir, comment='Demo')
        
        tb_metric_cb = partial(TBMetricCB, 
                               writer=writer,
                               train_metrics={'losses':['train_loss'], 'general':['lr', 'train_dice']}, 
                               validation_metrics={'losses':['val_loss'], 'general':['valid_dice', 'valid_dice2']})

        tb_predict_cb = partial(TBPredictionsCB, writer=writer, logger=logger, step=step)

        tb_cbs = [tb_metric_cb(), tb_predict_cb()]

        checkpoint_cb = CheckpointCB(models_dir, ema=cfg.TRAIN.EMA, save_step=cfg.TRAIN.SAVE_STEP)
        train_timer_cb = sh.callbacks.TimerCB(mode_train=True, logger=logger)
        master_cbs = [train_timer_cb, *tb_cbs, checkpoint_cb]
    
    l0,l1,l2, scale = cfg.TRAIN.LRS
    l0,l1,l2 = l0 * scale, l1 * scale, l2 * scale # scale if for DDP , cfg.PARALLEL.WORLD_SIZE
    l3, l4 = l0, l1 * .3
    l5, l6 = l0, l1 * 1

    lr_cos_sched = sh.schedulers.combine_scheds([
        [.2, sh.schedulers.sched_cos(l0,l1)],
        [.8, sh.schedulers.sched_cos(l1,l2)],
        ])
    lrcb = sh.callbacks.ParamSchedulerCB('before_epoch', 'lr', lr_cos_sched)

    cbs = [CudaCB(), train_cb, val_cb, lrcb]
        
    if cfg.PARALLEL.IS_MASTER:
        cbs.extend(master_cbs)
        # TODO nfold in epochbar
        epoch_bar = master_bar(range(n_epochs))
        batch_bar = partial(progress_bar, parent=epoch_bar)
    else: epoch_bar, batch_bar = range(n_epochs), lambda x:x

    logger.log("DEBUG", 'LEARNER') 
    val_interval = cfg.VALID.STEP
    learner = sh.learner.Learner(model, opt, sh.utils.AttrDict(dls), criterion, 0, cbs, batch_bar, epoch_bar, val_interval, cfg=cfg)
    learner.fit(n_epochs)

    # Just in case  
    gc.collect()
    torch.cuda.empty_cache()
    del dls 
    del learner
    del cbs
    gc.collect()
    torch.cuda.empty_cache()
