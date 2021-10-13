import os
import math
import json
import random
import argparse
import datetime
import itertools
from pathlib import Path
from functools import partial
import multiprocessing as mp
import multiprocessing.pool
from contextlib import contextmanager
from typing import Tuple, List, Dict, Callable

import cv2
import torch
import rasterio
import numpy as np
from shapely import geometry

from config import cfg


def jread(path: str) -> Dict:
    with open(str(path), 'r') as f:
        data = json.load(f)
    return data


def jdump(data: Dict, path: str) -> None:
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=4)


def filter_ban_str_in_name(s: str, bans: List[str]): return any([(b in str(s)) for b in bans])


def get_filenames(path: str, pattern: str, filter_out_func: Callable) -> str:
    """
    pattern : "*.json"
    filter_out : function that return True if file name is acceptable
    """

    filenames = list(Path(path).glob(pattern))
    print(filenames)
    assert (filenames), f'There is no matching filenames for {path}, {pattern}'
    filenames = [fn for fn in filenames if not filter_out_func(fn)]
    # assert (filenames), f'There is no matching filenames for {filter_out_func}'
    return filenames


def polyg_to_mask(polyg: np.ndarray, wh: Tuple[int, int], fill_value: int) -> np.ndarray:
    """Convert polygon to binary mask.
    """

    polyg = np.int32([polyg])
    mask = np.zeros([wh[0], wh[1]], dtype=np.uint8)
    cv2.fillPoly(mask, polyg, fill_value)
    return mask


def json_record_to_poly(record: Dict) -> List[geometry.Polygon]:
    """Get list of polygons from record.
    """

    num_polygons = len(record['geometry']['coordinates'])
    if num_polygons == 1:     # Polygon
        list_coords = [record['geometry']['coordinates'][0]]
    elif num_polygons > 1:    # MultiPolygon
        list_coords = [record['geometry']['coordinates'][i][0] for i in range(num_polygons)]
    else:
        raise Exception("No polygons are found")

    try:
        polygons = [geometry.Polygon(coords) for coords in list_coords]
    except Exception as e:
        print(e, list_coords)
    return polygons


def make_folders(cfg):
    cfg_postfix = 'PAMBUH'
    timestamp = '{:%Y_%b_%d_%H_%M_%S}'.format(datetime.datetime.now())
    name = Path(timestamp + '_' + cfg_postfix)
    fname = cfg.OUTPUTS / name
    fname.mkdir(exist_ok=True)
    return fname

def save_models(d, postfix, output_folder):
    for k, v in d.items():
        torch.save(v.state_dict(), output_folder/os.path.join("models",f"{k}_{postfix}.pkl"))

def dump_params(cfg, output_path):
    with open(os.path.join(output_path, 'cfg.yaml'), 'w') as f:
        f.write(cfg.dump())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    return args

def get_basics_rasterio(name):
    file = rasterio.open(str(name))
    return file, file.shape, file.count

def get_tiff_block(ds, x, y, w, h=None, bands=3):
    if h is None: h = w
    return ds.read(list(range(1, bands+1)), window=rasterio.windows.Window(x, y, w, h))

def save_tiff_uint8_single_band(img, path, bits=1):
    assert img.dtype == np.uint8
    if img.max() <= 1. : print(f"Warning: saving tiff with max value is <= 1, {path}")
    h, w = img.shape
    dst = rasterio.open(path, 'w', driver='GTiff', height=h, width=w, count=1, nbits=bits, dtype=np.uint8)
    dst.write(img, 1)
    dst.close()
    del dst

def cfg_frz(func):
    def frz(*args, **kwargs):
        cfg.defrost()
        r = func(*args, **kwargs)
        cfg.freeze()
        return r
    return frz

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def mp_func(foo, args, n):
    args_chunks = [args[i:i + n] for i in range(0, len(args), n)]
    with poolcontext(processes=n) as pool:
        res = pool.map(foo, args_chunks)
    return [ri for r in res for ri in r]


def mp_func_gen(foo, args, n, progress=None):
    args_chunks = [args[i:i + n] for i in range(0, len(args), n)]
    results = []
    with poolcontext(processes=n) as pool:
        gen = pool.imap(foo, args_chunks)
        if progress is not None: gen = progress(gen, total=len(args_chunks))
        for r in gen:
            results.extend(r)
    return results


def get_cortex_polygons(anot_structs_json: Dict) -> List[geometry.Polygon]:
    return get_polygons_by_type(anot_structs_json, 'Cortex')

def get_polygons_by_type(anot_structs_json: Dict, name: str) -> List[geometry.Polygon]:
    polygons = []
    for record in anot_structs_json:
        if record['properties']['classification']['name'] == name:
            polygons += json_record_to_poly(record)
    return polygons

def flatten_2dlist(list2d: List) -> List:
    """Converts 2d list into 1d list.
    """

    list1d = list(itertools.chain(*list2d))
    return list1d

def tiff_merge_mask(path_tiff, path_mask, path_dst, path_mask2=None):
    # will use shitload of mem
    img = rasterio.open(path_tiff).read()
    mask = rasterio.open(path_mask).read()
    #assert mask.max() <= 1 + 1e-6

    if img.shape[0] == 1:
        img = np.repeat(img, 3, 0)


    red = mask * 200 if mask.max() <= 1 + 1e-6 else mask
    img[1,...] = img.mean(0)
    img[0,...] = red

    if path_mask2 is not None:
        mask2 = rasterio.open(path_mask2).read()
        blue = mask2 * 200 if mask2.max() <= 1 + 1e-6 else mask2
        #assert mask2.max() <= 1 + 1e-6
        img[2,...] = blue

    _, h, w = img.shape
    dst = rasterio.open(path_dst, 'w', driver='GTiff', height=h, width=w, count=3, dtype=np.uint8)
    dst.write(img, [1,2,3]) # 3 bands
    dst.close()
    del dst


def gen_pt_in_poly(polygon: geometry.Polygon,
                   max_num_attempts=50) -> geometry.Point:
    """Generates randomly point within given polygon. If after max_num_attempts point has been not
    found, then returns centroid of polygon.
    """

    min_x, min_y, max_x, max_y = polygon.bounds

    num_attempts = 0
    while num_attempts < max_num_attempts:
        random_point = geometry.Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if random_point.within(polygon): return random_point
        num_attempts += 1
    return polygon.centroid


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Gets np.ndarray (3, ...) or (..., 3) and returns gray scale np.ndarray (...)."""

    first_channel = rgb.shape[0]
    if first_channel == 3:
        rgb = np.swapaxes(np.swapaxes(rgb, 0, 2), 0, 1) # (3, ...) -> (..., 3)
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def save_arr_as_tiff(arr: np.ndarray, path: str, nbits: int = 8) -> None:
    """Gets np.ndarray (num_bands, h, w) and returns gray scale np.ndarray (h, w) in uint8."""
    
    num_bands, h, w = arr.shape

    dst = rasterio.open(path, 'w', driver='GTiff',
                        height=h, width=w, count=num_bands,
                        nbits=nbits, dtype=np.uint8)
    dst.write(arr)
    dst.close()
    del dst


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonPool(mp.pool.Pool):
    Process = NoDaemonProcess

def sigmoid(x): return 1 / (1 + np.exp(-x))
