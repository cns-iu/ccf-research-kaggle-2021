import os
from PIL import Image
from pathlib import Path
from functools import partial, reduce
from collections import defaultdict
import multiprocessing as mp
from contextlib import contextmanager

import cv2
import numpy as np
from tqdm.auto import tqdm

import utils
import data
import sampler
from rle2tiff import start as create_masks
from split_gen import do_split


def cut_glomi(imgs_path, masks_path, dst_path):
    filt = partial(utils.filter_ban_str_in_name, bans=['-', '_ell'])
    masks_fns = sorted(utils.get_filenames(masks_path, '*.tiff', filt))
    img_fns = sorted(utils.get_filenames(imgs_path, '*.tiff', filt))
    ann_fns = sorted(utils.get_filenames(imgs_path, '*.json', filt))
    #print(img_fns, masks_fns)
    SCALE = 2
    _base_wh = 1024
    wh = (_base_wh*SCALE, _base_wh*SCALE)
    
    for i_fn, m_fn, a_fn in tqdm(zip(img_fns, masks_fns, ann_fns)):
        s = sampler.GdalSampler(i_fn, m_fn, a_fn, wh)
        
        img_dir = dst_path / 'imgs' / i_fn.with_suffix('').name
        os.makedirs(str(img_dir), exist_ok=True)

        mask_dir = dst_path / 'masks' / i_fn.with_suffix('').name
        os.makedirs(str(mask_dir), exist_ok=True)
        
        for idx, (i,m) in enumerate(s):      
        
            orig_name = (str(idx).zfill(6) + '.png')
            
            img_name = img_dir / orig_name 
            mask_name = mask_dir /orig_name

            i = i.transpose(1,2,0)
            m = m.transpose(1,2,0)
            
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            #m = m.repeat(3,-1).astype(np.uint8)
            m = 255 * m.repeat(3,-1).astype(np.uint8) # as our masks are one bit : 0-1
            
            i = cv2.resize(i, (wh[0]//SCALE, wh[1]//SCALE), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (wh[0]//SCALE, wh[1]//SCALE), interpolation=cv2.INTER_AREA)

            # print(i.shape(), m.shape())
            cv2.imwrite(str(img_name), i)
            cv2.imwrite(str(mask_name), m)

            print(idx)

            # DEBUG MODE:
            if idx > 35: break

def cut_grid(imgs_path, masks_path, dst_path):
    filt = partial(utils.filter_ban_str_in_name, bans=['-', '_ell'])
    img_fns = sorted(utils.get_filenames(imgs_path, '*.tiff', filt))
    masks_fns = sorted(utils.get_filenames(masks_path, '*.tiff', filt))

    SCALE = 2
    _base_wh = 1024
    wh = (_base_wh*SCALE, _base_wh*SCALE)

    print(img_fns, masks_fns)

    for i_fn, m_fn, in tqdm(zip(img_fns, masks_fns)):
        s = sampler.GridSampler(i_fn, m_fn, wh)
        
        img_dir = dst_path / 'imgs' / i_fn.with_suffix('').name
        os.makedirs(str(img_dir), exist_ok=True)

        mask_dir = dst_path / 'masks' / i_fn.with_suffix('').name
        os.makedirs(str(mask_dir), exist_ok=True)

        if len(s) == 0:
            print("NO GRID ", img_dir)
        
        for idx, (i,m) in enumerate(s):      
            if   (i.mean() < 10 or
                 (i.mean() > 205 and i.std() < 20 ) or 
                 ((i.mean(0)<10).sum() > 0.1 * wh[0] * wh[1])
                )  and (np.random.random() > .05):
                continue
            orig_name = (str(idx).zfill(6) + '.png')
            print(i.shape, i.mean(), i.std(), (i.mean(0)<10).sum() / (wh[0] * wh[1]) , orig_name)
            
            
            img_name = img_dir / orig_name 
            mask_name = mask_dir /orig_name
            
            i = i.transpose(1,2,0)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            
            m = m.transpose(1,2,0)
            m = 255 * m.repeat(3,-1).astype(np.uint8)
            
            i = cv2.resize(i, (wh[0]//SCALE, wh[1]//SCALE), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (wh[0]//SCALE, wh[1]//SCALE), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(str(img_name), i)
            cv2.imwrite(str(mask_name), m)

            print(idx)

            # DEBUG MODE:
            if idx > 35: break

if __name__  == "__main__":
    src = 'input/'
    src = Path(src)
    hub_src = src / 'HUBMAP'
    assert hub_src.exists()

    masks_path = src / 'bigmasks' # will be created
    try:
        masks_path.mkdir(exist_ok=False)
        create_masks(str(hub_src), str(masks_path))
    except FileExistsError:
        print('\n\nMASKS ALREADY CREATED? SKIPING BIG TIFF MASK CREATION')

    imgs_path = hub_src / 'train'
    grid_path = src / 'CUTS/grid_x33_1024/'
    glomi_path = src / 'CUTS/glomi_x33_1024/'

    # this cuts big tiffs based on annotation json, each object get its own cut
    if not glomi_path.exists(): cut_glomi(imgs_path, masks_path, glomi_path)
    # this cuts big tiffs on tile based grid
    if not grid_path.exists(): cut_grid(imgs_path, masks_path, grid_path)

    # This breaks cutted tiff's into train and val splits based in tiff ids
    do_split(grid_path, src / 'SPLITS/grid_split' )
    do_split(glomi_path, src / 'SPLITS/glomi_split' )

