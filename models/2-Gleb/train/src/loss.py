from __future__ import print_function, division
from functools import partial

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


class EdgeLoss:
    def __init__(self, mode='single', gpu=True):
        self.mode = mode
        self.device = 'cuda' if gpu else 'cpu'
        k1, k2 = 3, 15
        #self.get_edge_1 = partial(torch.nn.functional.conv2d, weight=torch.ones(1,1,k1,k1).to(device)/k1**2, padding=1) # TODO padding
        self.get_edge_2 = partial(torch.nn.functional.conv2d, weight=torch.ones(1,1,k2,k2).to(self.device)/k2**2, padding=(k2-1)//2)
        self.dog_thres = 1e-4
        self.loss = torch.nn.BCEWithLogitsLoss()
    
    def get_edges(self, yb):
        with torch.no_grad():
            # TODO: torchsript
            #dog = self.get_edge_1(yb) - self.get_edge_2(yb)
            dog = yb - self.get_edge_2(yb)
        return dog < -self.dog_thres, dog > self.dog_thres
    
    def __call__(self, pb, yb, **kwargs):
        l = torch.tensor(0.).to(self.device)
        bs = yb.shape[0]
        
        for y, p, e1, e2 in zip(yb, pb, *self.get_edges(yb.detach())):
            if not (e1.any() and e2.any()): continue
            if self.mode == 'single':
                # EE
                g  = torch.cat((y.reshape(1,-1)[0], y[e1 == 1], y[e2 == 1]), 0).unsqueeze(0)
                y_h= torch.cat((p.reshape(1,-1)[0], p[e1 == 1], p[e2 == 1]), 0).unsqueeze(0)
            elif self.mode == 'double':
                # Double EE
                g  = torch.cat((y.reshape(1,-1)[0], y[e1 == 1], y[e1 == 1], y[e2 == 1]), 0).unsqueeze(0)
                y_h= torch.cat((p.reshape(1,-1)[0], p[e1 == 1], p[e1 == 1], p[e2 == 1]), 0).unsqueeze(0)
            elif self.mode == 'edge':
                # Only edge
                g  = torch.cat((y[e1 == 1], y[e2 == 1]), 0).unsqueeze(0)
                y_h= torch.cat((p[e1 == 1], p[e2 == 1]), 0).unsqueeze(0)

            l += self.loss(y_h, g)

        l /= bs
        return l

def dice_loss(inp, target, eps=1e-6):
    with torch.no_grad():
        a = inp.contiguous().view(-1)
        b = target.contiguous().view(-1)
        intersection = (a * b).sum()
        dice = ((2. * intersection + eps) / (a.sum() + b.sum() + eps)) 
        return dice.item()

def focal_loss(outputs: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
    """
    Compute binary focal loss between target and output logits.

    Args:
        reduction (string, optional):
            none | mean | sum | batchwise_mean
            none: no reduction will be applied,
            mean: the sum of the output will be divided by the number of elements in the output,
            sum: the output will be summed.
    Returns:
        computed loss

    Source: https://github.com/BloodAxe/pytorch-toolbelt, Catalyst
    """
    targets = targets.type(outputs.type())
    logpt = -torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt
    #if alpha is not None: loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))

    if reduction == "mean": loss = loss.mean()
    if reduction == "sum": loss = loss.sum()
    if reduction == "batchwise_mean": loss = loss.sum(0)

    return loss

def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / (union)
    #print('jac: ', jaccard.max(), jaccard.min())
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        losses = (lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) for log, lab in zip(logits, labels))
        loss = mean(losses, ignore_nan=True)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    if not isinstance(loss, torch.Tensor):
        print('wtf')
        print(logits.shape, logits)
        print(loss, logits.max(), logits.min(), labels.max(), labels.min())
    elif torch.isnan(loss):
        print(loss, logits.max(), logits.min(), labels.max(), labels.min())
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    #loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(torch.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
