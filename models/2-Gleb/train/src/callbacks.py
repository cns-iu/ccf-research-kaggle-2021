import os
import time
import datetime
import pickle
from pathlib import Path
from functools import partial
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

import ema
import utils
import loss
import shallow as sh
from logger import logger


def upscale(tensor, size): return torch.nn.functional.interpolate(tensor, size=size)

def denorm(images, mean=(0.46454108, 0.43718538, 0.39618185), std=(0.23577851, 0.23005974, 0.23109385)):
    mean = torch.tensor(mean).view((1,3,1,1))
    std = torch.tensor(std).view((1,3,1,1))
    images = images * std + mean
    return images

def get_xb_yb(b):
    return b[0], b[1]

def get_pred(m, xb):
    r = m(xb)
    if isinstance(r, tuple): return r
    else: return r, None

def get_tag(b):
    if len(b) == 3: return b[2]
    else: return None

class CudaCB(sh.callbacks.Callback):
    def before_batch(self):
        xb, yb = get_xb_yb(self.batch)
        bb = get_tag(self.batch)

        yb = yb.cuda()
        if bb is None: self.learner.batch = xb.cuda(), yb
        else: self.learner.batch = xb.cuda(), yb, bb.cuda()

    def before_fit(self): self.model.cuda()

class TrackResultsCB(sh.callbacks.Callback):
    """
        TODO break alliance with TB metric CB
    """
    def before_epoch(self): 
        self.accs,self.losses,self.samples_count = [],[],[]
        
    def after_epoch(self):
        n = sum(self.samples_count)
        print(self.n_epoch, self.model.training, sum(self.losses)/n, sum(self.accs)/n)
        
    def after_batch(self):
        with torch.no_grad():
            xb, yb = get_xb_yb(self.batch)
            tag = get_tag(self.batch)
            batch_size = xb.shape[0]
            #print(self.preds.shape, yb.shape, xb.shape)
            p = torch.sigmoid(self.preds)
            p = (p>.5)
            dice = loss.dice_loss(p.float(), yb.float())
            #print(n, dice, dice*n)
            self.accs.append(dice * batch_size)
            self.samples_count.append(batch_size)
            self.losses.append(self.loss.detach().item()*batch_size)


class TBMetricCB(TrackResultsCB):
    def __init__(self, writer, train_metrics=None, validation_metrics=None, logger=None):
        ''' train_metrics = {'losses':['train_loss', 'val_loss']}
            val_metrics = {'metrics':['localization_f1']}
        '''
        sh.utils.store_attr(self, locals())
        self.max_dice = 0

    def parse_metrics(self, metric_collection):
        if metric_collection is None : return
        for category, metrics in metric_collection.items():
            for metric_name in metrics:
                metric_value = getattr(self, metric_name, None)
                if metric_value is not None: 
                    self.log_debug(f"{category + '/' + metric_name, metric_value, self.n_epoch}")
                    self.writer.add_scalar(category + '/' + metric_name, metric_value, self.n_epoch)


    def after_epoch_train(self):
        #self.log_debug('tb metric after train epoch')
        print(self.samples_count)
        if self.samples_count == []:
            self.samples_count = [1]
        self.train_loss = sum(self.losses) / sum(self.samples_count)
        self.train_dice =  sum(self.accs) / sum(self.samples_count)
        self.parse_metrics(self.train_metrics)
        
    def after_epoch_valid(self):
        #self.log_debug('tb metric after validation')
        self.val_loss = sum(self.losses) / sum(self.samples_count)
        self.valid_dice =  sum(self.accs) / sum(self.samples_count)
        self.valid_dice2 =  sum(self.learner.extra_accs) / sum(self.learner.extra_samples_count)
        self.parse_metrics(self.validation_metrics)

        save_val_threshold = .7 # Only save models that do DICE > 0.90
        if self.valid_dice > self.max_dice:
            self.max_dice = self.valid_dice
            if self.max_dice > save_val_threshold:
                chpt_cb = get_cb_by_instance(self.learner.cbs, CheckpointCB)
                if chpt_cb is not None: chpt_cb.do_saving(f'cmax_val_{round(self.max_dice, 4)}', save_ema=False)

        if self.valid_dice2 > self.max_dice:
            self.max_dice = self.valid_dice2
            if self.max_dice > save_val_threshold:
                chpt_cb = get_cb_by_instance(self.learner.cbs, CheckpointCB)
                if chpt_cb is not None: chpt_cb.do_saving(f'cmax_ema_{round(self.max_dice, 4)}', save_ema=True)
            
    def after_epoch(self):
        if self.model.training: self.after_epoch_train()
        else: self.after_epoch_valid()
        self.writer.flush()

        
class TBPredictionsCB(sh.callbacks.Callback):
    def __init__(self, writer, logger=None, step=1):
        sh.utils.store_attr(self, locals())
        self.count, self.wh = 5, (256, 256)

    def process_batch(self):
        xb, yb = get_xb_yb(self.batch)
        preds = self.preds
        num_channels = 1
        mean, std = self.kwargs['cfg'].TRANSFORMERS.MEAN, self.kwargs['cfg'].TRANSFORMERS.STD 

        xb = xb[:self.count].float().detach().cpu()
        xb = denorm(xb, mean, std)
        xb = upscale(xb, self.wh)

        yb = yb[:self.count].repeat(1,3,1,1)
        yb = upscale(yb.detach().cpu().float(), self.wh)

        preds = torch.sigmoid(preds[:self.count, ...])
        preds = preds.max(1, keepdim=True)[0].repeat(1,3,1,1)
        preds = upscale(preds.float().detach().cpu(), self.wh)

        return xb, yb, preds
    
    def process_write_predictions(self):
        #self.log_debug('tb predictions')
        xb, yb, preds = self.process_batch() # takes last batch that been used
        #self.log_debug(f"{xb.shape}, {yb.shape}, {preds.shape}")
        summary_image = torch.cat([xb,yb,preds])
        #self.log_debug(f"{summary_image.shape}")
        grid = torchvision.utils.make_grid(summary_image, nrow=self.count, pad_value=4)
        label = 'train predictions' if self.model.training else 'val_predictions'
        self.writer.add_image(label, grid, self.n_epoch)
        self.writer.flush()

    def after_epoch(self):
        if not self.model.training or self.n_epoch % self.step ==0:
            self.process_write_predictions()


class TrainCB(sh.callbacks.Callback):
    def __init__(self, logger=None): 
        sh.utils.store_attr(self, locals())
        self.cl_criterion = partial(torch.nn.functional.binary_cross_entropy_with_logits, reduction='none')
        self.cll = []

    def before_fit(self): self.freeze_enc = self.kwargs['cfg']["TRAIN"]["FREEZE_ENCODER"]

    @sh.utils.on_train
    def before_epoch(self):
        if self.kwargs['cfg'].PARALLEL.DDP: self.dl.sampler.set_epoch(self.n_epoch)
        if self.freeze_enc and self.np_epoch > .45:
            self.freeze_enc = False
            self.log_debug(f'UNFREEZING ENCODER at {self.np_epoch}')
            unwrap_model(self.model).encoder.requires_grad_(True)

        for i in range(len(self.opt.param_groups)):
            self.learner.opt.param_groups[i]['lr'] = self.lr  
        self.cl_gt = []
        self.cl_pred = []

    @sh.utils.on_train
    def after_epoch(self):
        pass

    def train_step(self):
        xb, yb = get_xb_yb(self.batch)
        tag = get_tag(self.batch) 
        self.learner.preds, aux_cl_pred = get_pred(self.model, xb)
        loss = self.loss_func(self.preds, yb)

        self.learner.loss = loss
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), .8)
        self.learner.loss.backward()
        self.learner.opt.step()
        self.learner.opt.zero_grad(set_to_none=True)

        if self.kwargs['cfg'].TRAIN.EMA: self.learner.model_ema.update(self.model)


class ValCB(sh.callbacks.Callback):
    def __init__(self, dl=None, logger=None):
        self.extra_valid_dl = dl
        sh.utils.store_attr(self, locals())
        self.evals = []
 
    @sh.utils.on_validation
    def before_epoch(self):
        if self.extra_valid_dl is not None:
            self.run_extra_valid()
        self.learner.metrics = []
        self.reduction = 'none' if self.kwargs['cfg'].TRAIN.SELECTIVE_BP <1 - 1e-6 else 'mean'

    def run_extra_valid(self):
        self.learner.extra_accs, self.learner.extra_samples_count = [], []
        for batch in self.extra_valid_dl:
            xb, yb = get_xb_yb(batch)
            tag = get_tag(batch)
            batch_size = xb.shape[0]

            with torch.no_grad():
                preds , _ = get_pred(self.model, xb)

            #print(preds.shape, yb.shape, xb.shape)
            p = torch.sigmoid(preds.cpu().float())
            p = (p>.5).float()
            dice = loss.dice_loss(p, yb.cpu().float())
            self.learner.extra_accs.append(dice * batch_size)
            self.learner.extra_samples_count.append(batch_size)

    def val_step(self):
        xb, yb = get_xb_yb(self.batch)
        with torch.no_grad():
            self.learner.preds, _ = get_pred(self.model, xb)
            self.learner.loss = self.loss_func(self.preds, yb)
                

class ValEMACB(sh.callbacks.Callback):
    def __init__(self, model_ema, logger=None):
        sh.utils.store_attr(self, locals())
        self.evals = []

    def before_fit(self):
        self.learner.model_ema = self.model_ema

    @sh.utils.on_validation
    def before_epoch(self):
        self.run_ema_valid()
        self.learner.metrics = []

    def run_ema_valid(self):
        self.learner.extra_accs, self.learner.extra_samples_count = [], []
        for batch in self.dls['VALID']:
            xb, yb = get_xb_yb(batch)
            tag = get_tag(batch)
            batch_size = xb.shape[0]

            with torch.no_grad():
                preds, _ = get_pred(self.learner.model_ema.module, xb.cuda())

            p = torch.sigmoid(preds.cpu().float())
            p = (p>.5).float()
            dice = loss.dice_loss(p, yb.float())
            self.learner.extra_accs.append(dice * batch_size)
            self.learner.extra_samples_count.append(batch_size)


    def val_step(self):
        xb, yb = get_xb_yb(self.batch)
        with torch.no_grad():
            self.learner.preds , _ = get_pred(self.model, xb)
            self.learner.loss = self.loss_func(self.preds, yb)
                

def get_cb_by_instance(cbs, cls):
    for cb in cbs:
        if isinstance(cb, cls): return cb
    return None

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()

class CheckpointCB(sh.callbacks.Callback):
    def __init__(self, save_path, ema=False, save_step=None):
        sh.utils.store_attr(self, locals())
        self.pct_counter = None if isinstance(self.save_step, int) else self.save_step
        
    def do_saving(self, val='', save_ema=True):
        m = self.model_ema if save_ema else self.model
        name = m.name if hasattr(m, 'name') else None
        state_dict =  get_state_dict(m) 
        torch.save({
                'epoch': self.n_epoch,
                'loss': self.loss,
                'model_state': state_dict,
                'opt_state': self.opt.state_dict(), 
                'model_name': name, 
            }, str(self.save_path / f'e{self.n_epoch}_t{self.total_epochs}_{val}.pth'))

    def after_epoch(self):
        save = False
        if self.n_epoch == self.total_epochs - 1: save=False
        elif isinstance(self.save_step, int): save = self.save_step % self.n_epoch == 0 
        else:
            if self.np_epoch > self.pct_counter:
                save = True
                self.pct_counter += self.save_step
        
        if save: self.do_saving('_AE')
