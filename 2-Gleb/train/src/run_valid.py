import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import datetime
from pathlib import Path
from logger import logger
from itertools import cycle
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio

import utils
from run_inference import start_inf
from loss import dice_loss


def get_split_by_idx(idx, use_split_a=True):
    imgs = Path('input/hm/train').glob('*.tiff')
    if use_split_a:
        splits = [
            ['0486052bb', 'e79de561c'],
            ['2f6ecfcdf', 'afa5e8098'],
            ['1e2425f28', '8242609fa'],
            ['cb2d976f4', 'c68fe75ea']]
    else:
        splits = [
                ['0486052bb', 'e79de561c','2f6ecfcdf', 'afa5e8098'],
                ['1e2425f28', '8242609fa','cb2d976f4', 'c68fe75ea'],
                ['aaa6a05cc', 'b2dc8411c', '095bf7a1f', '54f2eec69'],
                ['b9a3865fc', '26dc41664', '4ef6695ce']]

    return [i for i in imgs if i.stem in splits[idx]]

def valid_dice(a,b, eps=1e-6):
    intersection = (a * b).sum()
    dice = ((2. * intersection + eps) / (a.sum() + b.sum() + eps)) 
    return dice

def calc_all_dices(res_masks, thrs, use_torch):
    dices = {}
    for img_name, mask_name in res_masks.items():
        logger.log('DEBUG', f'Dice for {img_name}')
        gt = rio.open(f'input/masks/bigmasks/{img_name.name}').read()
        mask = np.load(mask_name)

        if use_torch:
            import torch
            os.environ['CUDA_VISIBLE_DEVICES'] = '3'
            gt = torch.from_numpy(gt)
            #mask = torch.from_numpy(mask).float()
            mask = torch.from_numpy(mask).half()

        name = img_name.stem

        ds = []
        gt = gt > 0
        if use_torch:
            gt = gt.cuda()
            mask = mask.cuda()

        for thr in tqdm(thrs):
            th_mask = (mask > thr)
            dice = valid_dice(gt, th_mask)
            if use_torch: dice = dice.item()
            ds.append(dice)

        ds = np.array(ds)
        print(name, ds.max(), thrs[np.argmax(ds)])
        dices[name] = ds
        del mask
        del gt
    return dices

def calc_common_dice(dices, thrs):
    best_thrs = []
    dd = []
    for k, v in dices.items():
        dd.append(v)

    dd = np.array(dd)
    thr = thrs[np.argmax(dd.sum(0))]
    #print(dd, dd.sum(0))
    return thr

#def get_thrs(): return np.arange(.2,.9, .025)
def get_thrs(): return np.arange(.2,.9, .01)

def get_stats(dices, thrs):
    df = pd.DataFrame(columns=['name', 'thr', 'score', 'real_thr', 'real_score'])

    targ = calc_common_dice(dices, thrs)
    best_idx = np.argmin((thrs - targ)**2)
    best_thr = thrs[best_idx]

    for i, (k, v) in enumerate(dices.items()):
        idx = np.argmax(v)
        df.loc[i] = ([k, thrs[idx], v[idx], best_thr, v[best_idx]])

    s = df.mean()
    s['name'] = 'AVE'
    df.loc[len(df)] = s
    return df

def join_totals(model_folder):
    totals = list(Path(model_folder).rglob('total.csv'))
    print(totals)

    dfs = []
    for t in totals:
        df = pd.read_csv(str(t), index_col=0)
        df = df.loc[:len(df)-2] 
        dfs.append(df)

    df = pd.concat(dfs)
    df = df[df['name'] != 'afa5e8098']
    s = df.mean()
    s['name'] = 'AVE'
    df.loc[len(df)] = s
    print(df)
    df = df.round(5)
    df.to_csv(f'{model_folder}/total_stats.csv')

def start_valid(model_folder, split_idx, do_inf, merge, gpus, use_torch, do_dice):
    os.makedirs(f'{model_folder}/predicts/masks', exist_ok=True)
    threshold = 0
    num_processes = len(gpus)
    save_predicts=True
    use_tta=True
    to_rle=False
    timestamped = False
    use_split_a=False

    img_names = get_split_by_idx(split_idx, use_split_a)
    if do_inf:
        res_masks = start_inf(model_folder, img_names, gpus, num_processes, use_tta, threshold, save_predicts, to_rle)
    else:
        res_masks = {}
        for i in img_names:
            res_masks[i] = model_folder/'predicts/raw'/(i.stem + '.npy')
        print(res_masks)

    logger.log('DEBUG', 'Predicts done')


    if do_dice:
        thrs = get_thrs()
        dices = calc_all_dices(res_masks, thrs, use_torch=use_torch)
        df_result = get_stats(dices, thrs).round(5)
        print(df_result)
        if timestamped:
            timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
            total_name = f'{model_folder}/total_{timestamp}.csv'
        else:
            total_name = f'{model_folder}/total.csv'
        df_result.to_csv(total_name)
        logger.log('DEBUG', f'Total saved {total_name}')

    if merge:
        os.makedirs(f'{model_folder}/predicts/combined', exist_ok=True)
        logger.log('DEBUG', 'Merging masks')
        for k, v in res_masks.items():
            mask_name1 = f'input/masks/bigmasks/{k.name}'
            mask_name2 = f'{model_folder}/predicts/masks/{k.name}'
            merge_name = f'{model_folder}/predicts/combined/{k.name}'
            logger.log('DEBUG', f'Merging {merge_name}')
            utils.tiff_merge_mask(k, mask_name1, merge_name, mask_name2)


if __name__ == '__main__':
    model_folder = Path('output/2021_May_10_01_07_51_PAMBUH/fold_0')
    FOLDS = 0#[0,1,2,3] # or int 
    do_inf = False
    do_dice = not do_inf
    use_torch = not do_inf
    merge = False
    #gpus = [2,3]
    gpus = [0,1,2,3]

    if isinstance(FOLDS, list):
        for split_idx in FOLDS:
            fold_folder = model_folder / f'fold_{split_idx}' 
            start_valid(fold_folder, split_idx, do_inf, merge, gpus, use_torch, do_dice)
        join_totals(model_folder)
    else:
        start_valid(model_folder, FOLDS, do_inf, merge, gpus, use_torch, do_dice)
    

