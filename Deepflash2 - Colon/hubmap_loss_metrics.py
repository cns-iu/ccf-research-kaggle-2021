import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from fastai.torch_core import TensorBase, flatten_check
from fastai.metrics import Metric
from fastai.metrics import Dice as FastaiDice



class TorchLoss(_Loss):
    'Wrapper class around loss function for handling different tensor types.'
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def _contiguous(self, x): return TensorBase(x.contiguous())

    def forward(self, *input):
        input = map(self._contiguous, input)
        return self.loss(*input) #

# from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/joint_loss.py
class WeightedLoss(_Loss):
    '''
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    '''
    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight

class JointLoss(_Loss):
    'Wrap two loss functions into one. This class computes a weighted sum of two losses.'

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)
    
    
# Multiclass metrics
    
class Dice(FastaiDice):
    "Dice coefficient metric for binary target in segmentation"
    def accumulate(self, learn):
        pred,targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        pred, targ  = map(TensorBase, (pred, targ))
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()

class Iou(Dice):
    "Implemetation of the IoU (jaccard coefficient) that is lighter in RAM"
    @property
    def value(self): return self.inter/(self.union-self.inter) if self.union > 0 else None
    
    
class Recall(Metric):
    def __init__(self, axis=1, th=0.5, epsilon=1e-7): 
        self.axis = axis 
        self.epsilon = epsilon
        self.th = th
    def reset(self): self.tp,self.fn = 0,0
    def accumulate(self, learn):
        pred,targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        self.tp += (pred*targ).float().sum().item()
        self.fn += (targ * (1 - pred)).float().sum().item()
    @property
    def value(self): return self.tp / (self.tp + self.fn + self.epsilon)
    
    
class Precision(Metric):
    def __init__(self, axis=1, th=0.5, epsilon=1e-7): 
        self.axis = axis 
        self.epsilon = epsilon
        self.th = th
    def reset(self): self.tp,self.fp = 0,0
    def accumulate(self, learn):
        pred,targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        self.tp += (pred*targ).float().sum().item()
        self.fp += ((1-targ) * pred).float().sum().item()
    @property
    def value(self): return self.tp / (self.tp + self.fp + self.epsilon)
    

    
