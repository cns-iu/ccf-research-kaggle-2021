# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,shallow//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
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
import torch
import torch.nn as nn

from shallow import utils


# %% [markdown]
# # Code

# %%
class CancelFitException(Exception): pass

class Learner:
    def __init__(self, model, opt, dls, loss_func, lr, cbs, batch_bar, epoch_bar, val_rate=1, **kwargs):
        utils.store_attr(self, locals())
        for cb in self.cbs: cb.learner = self

    def one_batch(self):
        self('before_batch')
        if self.model.training: self('train_step')
        else: self('val_step')
        self('after_batch')

    def one_epoch(self, train):
        self.model.training = train
        self.dl = self.dls.TRAIN if train else self.dls.VALID
        self('before_epoch')
        for self.n_batch, self.batch in enumerate(self.batch_bar(self.dl)):
            self.np_batch = self.n_batch / len(self.dl)
            self.one_batch()
        self('after_epoch')
    
    def fit(self, total_epochs):
        self('before_fit')
        self.total_epochs = total_epochs
        try:
            for self.n_epoch in self.epoch_bar:
                self.np_epoch = self.n_epoch / self.total_epochs
                self.one_epoch(True)
                if self.n_epoch % self.val_rate == 0: self.one_epoch(False)
                #self.progress_bar.master_bar.write(f'Finished loop {self.epoch}.')
        except CancelFitException: pass
        self('after_fit')
        
    def __call__(self, name):
        for cb in self.cbs: getattr(cb, name, utils.noop)()

# %%

# %%
