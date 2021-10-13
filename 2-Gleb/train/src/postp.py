import os
import sys
import json
import time
import random
from pathlib import Path
import multiprocessing as mp
from functools import partial
from logger import logger

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio

import utils
from tf_reader import TFReader
import rle2tiff

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def gpu_select(gpus):
    if isinstance(gpus, list): return gpus[0] # local run

    # MP run from run_inference
    time.sleep(2*random.random()) # syncing GPU dict between processes
    gpu = None
    while True:
        time.sleep(1)
        for k,v in gpus.items():
            if not v:
                gpu = k
                gpus[k] = True
                break
        if gpu is not None: break
    return gpu

def _count_blocks(dims, block_size):
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks

def get_basics_rasterio(name):
    file = rio.open(str(name))
    return file, file.shape, file.count

def generate_block_coords(H, W, block_size):
    h,w = block_size
    nYBlocks = (int)((H + h - 1) / h)
    nXBlocks = (int)((W + w - 1) / w)
    
    for X in range(nXBlocks):
        cx = X * h
        for Y in range(nYBlocks):
            cy = Y * w
            yield cy, cx, h, w

def pad_block(y,x,h,w, pad): return np.array([y-pad, x-pad, h+2*pad, w+2*pad])
def crop(src, y,x,h,w): return src[..., y:y+h, x:x+w]
def crop_rio(p, y,x,h,w):
    #ds = rio.open(str(p))
    ds = TFReader(str(p))
    block = ds.read(window=((y,y+h),(x,x+w)), boundless=True)
    del ds
    return block

def paste(src, block, y,x,h,w):src[..., y:y+h, x:x+w] = block
def paste_crop(src, part, block_cd, pad):
    _,H,W = src.shape
    y,x,h,w = block_cd
    h, w = min(h, H-y), min(w, W-x)  
    part = crop(part, pad, pad, h, w)
    paste(src, part, *block_cd)

def infer_blocks(blocks, do_inference):
    blocks = do_inference(blocks).cpu().numpy()    
    if isinstance(blocks, tuple): blocks = blocks[0]
    #return (255*blocks).astype(np.uint8)
    return blocks.astype(np.float16)
 
def image_q_reader(q, img_name, cds, pad):
    for block_cd in cds:
        padded_block_cd = pad_block(*block_cd, pad)
        block = crop_rio(img_name, *(padded_block_cd))
        #if block.shape[0] == 1: block = np.repeat(block, 3, 0)
        q.put((block, block_cd))
        
def mp_func_wrapper(func, args): return func(*args)
def chunkify(l, n): return [l[i:i + n] for i in range(0, len(l), n)]

def dump_to_csv(results, dst_path, test_folder, threshold):
    df = pd.read_csv(str(test_folder / 'sample_submission.csv'), index_col='id')
    for k, v in results.items():
        df.loc[k.stem] = v
    df.to_csv(str(dst_path / f'submission__{round(threshold, 3)}.csv'))


def mask_q_writer(q, H, W, batch_size, total_blocks, root, pad, use_tta, result):
    import infer

    do_inference = infer.get_infer_func(root, use_tta=use_tta)
    #mask = np.zeros((1,H,W)).astype(np.uint8)
    mask = np.zeros((1,H,W)).astype(np.float16)
    count = 0
    batch = []
    
    while count < total_blocks:
        if len(batch) == batch_size:
            blocks = [b[0] for b in batch]
            block_cds = [b[1] for b in batch]

            block_masks = infer_blocks(blocks, do_inference)
            [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, block in zip(block_masks, block_cds, blocks)]
            batch = []
        else:
            batch.append(q.get())
            count+=1
            q.task_done()

    if batch:
        blocks = [b[0] for b in batch]
        block_cds = [b[1] for b in batch]
        block_masks = infer_blocks(blocks, do_inference)
        [paste_crop(mask, block_mask, block_cd, pad) for block_mask, block_cd, block in zip(block_masks, block_cds, blocks) if block.mean() > 0]
        #print('Drop last batch', len(batch))
        batch = []
    
    print(mask.shape, mask.dtype, mask.max(), mask.min())

    img_name = result.keys()[0]
    raw_name = root / 'predicts/raw' / (img_name.stem + '.npy')
    os.makedirs(str(raw_name.parent), exist_ok=True)
    np.save(raw_name, mask)
    result[img_name] = raw_name


def launch_mpq(img_name, model_folder, batch_size, block_size, pad, num_processes, qsize, use_tta):
    m = mp.Manager()
    result = m.dict()
    result[img_name] = None
    q = m.Queue(maxsize=qsize)
    _, (H,W), _ = get_basics_rasterio(img_name)
    cds = list(generate_block_coords(H, W, block_size=(block_size,block_size)))
    #cds = cds[:20]
    total_blocks = len(cds)
        
    reader_args = [(q, img_name, part_cds, pad) for part_cds in chunkify(cds, num_processes)]
    reader = partial(mp_func_wrapper, image_q_reader)
    
    writer = partial(mp_func_wrapper, mask_q_writer)
    writer_p = mp.Process(target=writer, args=((q,H,W, batch_size, total_blocks, model_folder, pad, use_tta, result),))
    writer_p.start()        
    
    with mp.Pool(num_processes) as p:    
        g = tqdm(p.imap_unordered(reader, reader_args), total=len(reader_args))
        for _ in g:
            pass
    
    writer_p.join()
    writer_p.terminate()
    return  result[img_name]

            
def start(img_name, gpus, model_folder, results, use_tta=True):
    # params for inference, V100 32GB probably required for huge patches / batches
    scale = 3
    block_size = 1024 * scale
    batch_size = 8
    pad = 256 * scale
    num_processes = 8
    qsize = 24

    random.seed(hash(str(img_name)))
    gpu = gpu_select(gpus)
    logger.log('DEBUG', f'Starting inference {img_name} on GPU {gpu}')
    os.environ['CPL_LOG'] = '/dev/null'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


    raw_name = launch_mpq(img_name,
                        model_folder, 
                        batch_size=batch_size, 
                        block_size=block_size, 
                        pad=pad, 
                        num_processes=num_processes, 
                        qsize=qsize,
                        use_tta=use_tta)

    results[img_name] = raw_name
    if not isinstance(gpus, list): gpus[gpu] = False # Release gpu idx after usage

if __name__ == '__main__':
    # Single process run, mp in run_inference
    print(f'use run_inference.py')


def filter_mask(mask, name):
    with open(str(name.with_suffix(''))+ '-anatomical-structure.json', 'r') as f:
        data = json.load(f)
    h,w = mask.shape
    cortex_rec = [r for r in data if r['properties']['classification']['name'] == 'Cortex']
    cortex_poly = np.array(cortex_rec[0]['geometry']['coordinates']).astype(np.int32)
    buf = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(buf, cortex_poly, 1)
    mask = mask * buf
    return mask
