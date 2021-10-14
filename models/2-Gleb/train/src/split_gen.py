import os
import shutil
from pathlib import Path
from functools import partial
from datetime import datetime
from collections import defaultdict

import numpy as np

import utils


def get_images(root):
    imgs = list((root/'imgs').glob('*'))
    for img in imgs:
        filenames = list(img.glob('*'))
        yield filenames
        
def get_maskname_for_img(img_name):
    im_root = img_name.parent.parent
    mask_name  = img_name.parent.parent.parent / 'masks' / img_name.relative_to(im_root)
    return mask_name

def create_split(filenames, pct=.05):
    n = int(len(filenames) * pct)
    split = np.random.choice(filenames, n, replace=False).tolist()
    main_part = [f for f in filenames if f not in split]
    return main_part,  split

def create_split_from_polys(filenames, split_names):
    a,b = [], []
    for f in filenames:
        x = b if f.name in split_names else a
        x.append(f)
    return a,b

def select_samples_from_polys(name):
    data = utils.jread(Path(f'input/split_jsons/{name}.json'))
    val_poly = utils.json_record_to_poly(data[0])[0]
    
    glo_json = Path(f'input/hm/train/{name}.json')
    val_names = []
    
    data = utils.jread(glo_json)
    cnt = 0
    for i,d in enumerate(data):
        p = utils.json_record_to_poly(d)[0]
        if val_poly.contains(p.centroid) and cnt < 20:
            cnt += 1
            val_names.append(str(i).zfill(6) + '.png')
    print(cnt)
    return val_names

def copy_split(split, root, dst_path):
    p = dst_path / split.relative_to(root)
    os.makedirs(str(p.parent), exist_ok=True)
    shutil.copy(str(split), str(p))
    
def create_save_splits(root, dst_path, split_pct=None):
    '''
        takes root folder path with 2 folders inside: imgs, masks.
        for each subfolder in imgs, masks , i.e. 1e2425f28:
            splits images in subfolder in two groups randomly by split_pct:
            split_pct = 0.05
            len(p1) == .95 * len(p)
            len(p2) == .05 * len(p)
        and saves them into dst_path WITH TIMESTAMP 
        p1 is train folder, p2 is val folder
    '''
    for img_cuts in get_images(root):
        print(img_cuts[0].parent.name)
        
        if split_pct is not None:
            print('splitting randomly by percent')
            split_imgs_1, split_imgs_2 = create_split(img_cuts, pct=val_pct)
        else:
            print('splitting by predefined polygons in input/split_jsons')
            split_names = select_samples_from_polys(img_cuts[0].parent.name)
            print('selected:', split_names)
            split_imgs_1, split_imgs_2 = create_split_from_polys(img_cuts, split_names)
            
            
        print(len(split_imgs_1), len(split_imgs_2))

        for i in split_imgs_1:
            m = get_maskname_for_img(i)
            copy_split(i, root, dst_path/'train')
            copy_split(m, root, dst_path/'train')

        for i in split_imgs_2:
            m = get_maskname_for_img(i)
            copy_split(i, root, dst_path/'val')
            copy_split(m, root, dst_path/'val')

def do_split(src, dst):
    split_stems = [
         ['CL_HandE_1234_B004_topleft'],
         ['CL_HandE_1234_B004_topright'],
         ['HandE_B005_CL_b_RGB_bottomright'],
         ['HandE_B005_CL_b_RGB_topleft'],
         ]

    # split_stems = [
    #      ['CL_HandE_1234_B004_bottomright', 'HandE_B005_CL_b_RGB_bottomright'],
    #      ['CL_HandE_1234_B004_topleft', 'HandE_B005_CL_b_RGB_topleft'],
    #      ['CL_HandE_1234_B004_topright', 'HandE_B005_CL_b_RGB_bottomright'],
    #      ['CL_HandE_1234_B004_bottomright', 'HandE_B005_CL_b_RGB_topleft'],
    #      ]

    root = Path(src)
    dst = Path(dst)

    filt = partial(utils.filter_ban_str_in_name, bans=['-', '_ell', '_sc'])
    masks_fns = sorted(utils.get_filenames(root / 'masks', '*', filt))
    img_fns = sorted([m.parent.parent/'imgs'/m.name for m in masks_fns])

    for split in split_stems:
        name = split[0][0] + str(split_stems.index(split))
        path = dst / name
        train_path, val_path = path/'train', path/'val'
        os.makedirs(str(path))
        os.makedirs(str(train_path))
        os.makedirs(str(val_path))
        
        for imgs, masks in zip(img_fns, masks_fns):
        
            if imgs.stem in split:
                dst_path = val_path
            else:
                dst_path = train_path
                
            
            imgs_dst = dst_path / 'imgs' /  imgs.stem
            masks_dst = dst_path / 'masks' /  imgs.stem
            
            shutil.copytree(imgs, imgs_dst)
            shutil.copytree(masks, masks_dst)

