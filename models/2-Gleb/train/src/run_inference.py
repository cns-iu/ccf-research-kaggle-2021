import os
import time
import argparse
from pathlib import Path
from logger import logger
from itertools import cycle
from functools import partial
import multiprocessing as mp
import multiprocessing.pool

import numpy as np
from tqdm import tqdm

from postp import start, dump_to_csv, mp_func_wrapper
import utils
import rle2tiff



def start_inf(model_folder, img_names, gpu_list, num_processes, use_tta, thresholds, save_predicts=False, to_rle=False):
    logger.log('DEBUG', '\n'.join(list([str(i) for i in img_names])))

    m = mp.Manager()
    results = m.dict()
    gpus = m.dict()
    for i in gpu_list:
        gpus[i] = False

    starter = partial(start, model_folder=model_folder, gpus=gpus, results=results, use_tta=use_tta,)
    starter = partial(mp_func_wrapper, starter)
    args = [(name,) for name in img_names]

    with utils.NoDaemonPool(num_processes) as p:    
        g = tqdm(p.imap(starter, args), total=len(img_names))
        for _ in g:
            pass

    result_masks = dict(results)

    for img_name, mask_path in result_masks.items():
        mask = np.load(mask_path)[0]
        if save_predicts:
            out_name = model_folder/'predicts/masks'/img_name.name
            os.makedirs(str(out_name.parent), exist_ok=True)
            utils.save_tiff_uint8_single_band((255 * mask).astype(np.uint8), str(out_name), bits=8)

            logger.log('DEBUG', f'{img_name} done')
            if to_rle:
                logger.log('DEBUG', f'RLE...')
                threshold = thresholds[0]
                mask = (mask > threshold).astype(np.uint8)
                rle = rle2tiff.mask2rle(mask)
                result_masks[img_name] = rle

    return result_masks 

def read_results(model_folder, img_names, thresholds):
    mask_names = [model_folder / 'predicts/raw' / m.with_suffix('.npy').name for m in img_names]
    result_masks = {k:v for k,v in zip(img_names, mask_names)}

    for img_name, mask_path in result_masks.items():
        mask = np.load(mask_path)[0]
        logger.log('DEBUG', f'{img_name} done')
        logger.log('DEBUG', f'RLE...')
        threshold = thresholds[0]
        mask = (mask > threshold).astype(np.uint8)
        rle = rle2tiff.mask2rle(mask)
        result_masks[img_name] = rle

    return result_masks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--only_rle", const=True, default=True, nargs='?')
    parser.add_argument("--test_folder", default='input/hm/test', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
        Multiprocessing inference, one image per process / GPU
        Inference params in postp.py (BS, crop_size, queue size, etc)
        Results will be generated inside model_folder, in folder "predicts"
        Results are: 
            masks (uint8 0-255)
            npy masks (float16, 0-1)
            rle

    """
    args = parse_args()
    model_folder = Path(args.model_folder) #'output/2021_May_08_17_21_59_PAMBUH/'
    test_folder = Path(args.test_folder)
    do_inf = not args.only_rle # Do the prediction part, or just assemble results in rle

    gpu_list = [0]#[0,1,2,3]
    thresholds = [.5]
    use_tta = True
    save_predicts = True
    to_rle = True
    num_processes = len(gpu_list)

    img_names = list(Path(test_folder).glob('*.tiff'))
    print(img_names)

    if do_inf:
        results = start_inf(model_folder, 
                            img_names, 
                            gpu_list, 
                            num_processes, 
                            use_tta, 
                            thresholds=thresholds, save_predicts=save_predicts, to_rle=to_rle)
    else:
        results = read_results(model_folder, img_names, thresholds)

    if to_rle: dump_to_csv(results, model_folder, test_folder, thresholds[0])
