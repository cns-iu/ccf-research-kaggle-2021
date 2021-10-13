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

# %% tags=["active-ipynb"]
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %% [markdown]
# # Imports

# %%
import time
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from shallow import utils, meters


# %% [markdown]
# # Code

# %%
class CancelFitException(Exception): pass

class Callback(utils.GetAttr): 
    _default='learner'
    logger=None
    def log_debug(self, m): self.logger.log("DEBUG", m) if self.logger is not None else False
    def log_info(self, m): self.logger.log("INFO", m) if self.logger is not None else False
    def log_critical(self, m): self.logger.log("CRITICAL", m) if self.logger is not None else False
    
class ParamSchedulerCB(Callback):
    def __init__(self, phase, pname, sched_func):
        self.pname, self.sched_func = pname, sched_func
        setattr(self, phase, self.set_param)
        
    def set_param(self):
        setattr(self.learner, self.pname, self.sched_func(self.np_epoch))
    
class SetupLearnerCB(Callback):
    def before_batch(self):
        xb,yb = to_device(self.batch)
        self.learner.batch = tfm_x(xb),yb

    def before_fit(self): self.model.cuda()

class TrackResultsCB(Callback):
    def before_epoch(self): self.accs,self.losses,self.samples_count = [],[],[]
        
    def after_epoch(self):
        n = sum(self.ns)
        print(self.n_epoch, self.model.training, sum(self.losses)/n, sum(self.accs)/n)
        
    def after_batch(self):
        xb, yb = self.batch
        n = xb.shape[0]
        acc = (self.preds.argmax(dim=1)==yb).float().sum().item()
        self.accs.append(acc)
        self.samples_count.append(n)

        if self.model.training:
            self.losses.append(self.loss.detach().item()*n)

class LRFinderCB(Callback):
    def before_fit(self):
        self.losses,self.lrs = [],[]
        self.learner.lr = 1e-6
            
    def before_batch(self):
        if not self.model.training: return
        self.learner.lr *= 1.2
        print(self.lr)
        
    def after_batch(self):
        if not self.model.training: return
        if self.lr > 1 or torch.isnan(self.loss): raise CancelFitException
        self.losses.append(self.loss.item())
        self.lrs.append(self.lr)
        
        
class TimerCB(Callback):
    def __init__(self, Timer=None, mode_train=False, logger=None):
        self.logger = logger
        self.mode_train = mode_train
        self.perc=90
        if Timer is None: Timer = meters.StopwatchMeter
        self.batch_timer = Timer()
        self.epoch_timer = Timer()
    
    def _before_batch(self): 
        if self.model.training == self.mode_train: self.batch_timer.start()
    def before_epoch(self):
        if self.model.training == self.mode_train: self.epoch_timer.start()
    def _after_batch(self):
        if self.model.training == self.mode_train: self.batch_timer.stop()

    def after_epoch(self):
        if self.model.training == self.mode_train:
            self.epoch_timer.stop()
            bs, es = self.learner.dl.batch_size, len(self.learner.dl)
            self.log_info(f'\t[E {self.n_epoch}/{self.total_epochs}]: {self.epoch_timer.last: .3f} s,'+
                     f'{bs * es/self.epoch_timer.last: .3f} im/s; ')
                     #f'batch {self.batch_timer.avg: .3f} s'   )    
            self.batch_timer.reset()
    
    def after_fit(self):
        if self.model.training == self.mode_train:
            et = self.epoch_timer
            em = et.avg
            estd = ((et.p(self.perc) - em) + (em - et.p(1-self.perc))) / 2
            self.log_info(f'\tEpoch average time: {em: .3f} +- {estd: .3f} s')
        
        
class CheckpointCB(Callback):
    def __init__(self, save_path, save_step=None):
        utils.store_attr(self, locals())
        self.pct_counter = None if isinstance(self.save_step, int) else self.save_step
        
    def after_epoch(self):
        save = False
        if self.n_epoch == self.total_epochs - 1: save=True
        elif isinstance(self.save_step, int): save = self.save_step % self.n_epoch == 0 
        else:
            if self.np_epoch > self.pct_counter:
                save = True
                self.pct_counter += self.save_step
        
        if save:
            torch.save({
                            'epoch': self.n_epoch,
                            'loss': self.loss,
                            'model_state':self.model.state_dict(),
                            'opt_state':self.opt.state_dict(),                    
                        }, str(self.save_path / f'e{self.n_epoch}_t{self.total_epochs}_1e4l{int(1e4*self.loss)}.pth'))
            
            
class HooksCB(Callback):
    def __init__(self, func, layers, perc_start=.5, step=1, logger=None):
        utils.store_attr(self, locals()) 
        self.hooks = Hooks(self.layers, self.func)
        self.do_once = True
    
    @utils.on_epoch_step
    def before_batch(self):
        if self.do_once and self.np_batch > self.perc_start:
            self.log_debug(f'Gathering activations at batch {self.np_batch}')
            self.do_once = False
            self.hooks.attach()
    
    @utils.on_epoch_step
    def after_batch(self):
        if self.hooks.is_attached(): self.hooks.detach()
    @utils.on_epoch_step
    def after_epoch(self): self.do_once = True


# %%
class Hook():
    def __init__(self, m, f): self.m, self.f = m, f
    def attach(self): self.hook = self.m.register_forward_hook(partial(self.f, self))
    def detach(self): 
        if hasattr(self, 'hook') :self.hook.remove()
    def __del__(self): self.detach()
        
class Hooks(utils.ListContainer):
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args):
        self.attach()
        return self
    def __exit__ (self, *args): self.detach()
    def __del__(self): self.detach()

    def __delitem__(self, i):
        self[i].detach()
        super().__delitem__(i)
    
    def attach(self):
        for h in self: h.attach()
        
    def detach(self):
        for h in self: h.detach()
    
    def is_attached(self): return hasattr(self[0], 'hook')
            
def get_layers(model, conv=False, convtrans=False, lrelu=False, relu=False, bn=False, verbose=False):
    layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d): 
            if conv: layers.append(m)
        if isinstance(m, torch.nn.ConvTranspose2d): 
            if convtrans: layers.append(m)
        elif isinstance(m, torch.nn.LeakyReLU): 
            if lrelu: layers.append(m)
        elif isinstance(m, torch.nn.ReLU):
            if relu: layers.append(m)
        elif isinstance(m, torch.nn.BatchNorm2d):
            if bn: layers.append(m)
        else:
            if verbose: print(m)
    return layers
            
def append_stats(hook, mod, inp, outp, bins=100, vmin=0, vmax=0):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    means,stds,hists = hook.stats
    means.append(outp.data.mean().cpu())
    stds .append(outp.data.std().cpu())
    hists.append(outp.data.cpu().histc(bins,vmin,vmax))
        
def append_stats_buffered(hook, mod, inp, outp, device=torch.device('cpu'), bins=100, vmin=0, vmax=0):
    if not hasattr(hook,'stats'): hook.stats = (utils.TorchBuffer(shape=(1,), device=device),
                                                utils.TorchBuffer(shape=(1,), device=device),
                                                utils.TorchBuffer(shape=(bins,), device=device)
                                               )
    means,stds,hists = hook.stats
    means.push(outp.data.mean())
    stds .push(outp.data.std())
    hists.push(outp.data.float().histc(bins,vmin,vmax))


# %% [markdown]
# # Tests

# %% [markdown]
# ### Hooks

# %% tags=["active-ipynb"]
# model = torch.nn.Sequential(torch.nn.Conv2d(3,5,1), torch.nn.Conv2d(5,1,1))
#
# hooks = Hooks(model, append_stats_buffered)
#
# inp = torch.zeros(5,3,16,16)

# %% tags=["active-ipynb"]
# with hooks:
#     r = model(inp)
#

# %% tags=["active-ipynb"]
# stats = [h.stats for h in hooks]

# %%

# %% [markdown]
# ### Param sched

# %% tags=["active-ipynb"]
# import torch
# import matplotlib.pyplot as plt
#
# from shallow import schedulers

# %% tags=["active-ipynb"]
# class Learner:
#     def __init__(self, cbs):
#         self.cbs = cbs
#         self.lr = -1
#         self.total_epochs = 100
#         self.n_epoch = 13
#         for c in self.cbs: c.learner=self
#     
#     def t(self):
#         self('before_epoch')
#         
#     def tt(self):
#         self('bla_bla')
#         
#     def __call__(self, name):
#         for cb in self.cbs: getattr(cb, name, utils.noop())()

# %% tags=["active-ipynb"]
#
# p = ParamScheduler('before_epoch', 'lr', schedulers.sched_lin(0,.1))

# %% tags=["active-ipynb"]
# l = Learner([p])
# l.n_epoch = 88 # out of 100
# l.t()
# l.lr

# %% [markdown]
# ### Hooks

# %% [markdown]
# ## TB

# %% tags=["active-ipynb"]
# writer = SummaryWriter(comment='Demo')
# mycallback = partial(TensorBoardCB, writer, track_weight=True, track_grad=True, metric_names=['val loss', 'accuracy'])
#

# %%

# %%

# %%

# %%
