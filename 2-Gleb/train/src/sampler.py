import random 
import numpy as np
from PIL import Image
from typing import List, Tuple
from functools import partial

import rasterio
from shapely import geometry
from rasterio.windows import Window
from tf_reader import TFReader


from utils import jread, get_basics_rasterio, json_record_to_poly, flatten_2dlist, get_cortex_polygons, gen_pt_in_poly


class GdalSampler:
    """Iterates over img with annotation, returns tuples of img, mask
    """

    def __init__(self, img_path: str,
                 mask_path: str,
                 img_polygons_path: str,
                 img_wh: Tuple[int, int],
                 border_path=None,
                 rand_shift_range: Tuple[int, int] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """If rand_shift_range ~ (0,0), then centroid of glomerulus corresponds centroid of output sample
        """
        self._records_json = jread(img_polygons_path)
        self._mask = TFReader(mask_path)
        self._img = TFReader(img_path)
        self._border = TFReader(border_path) if border_path is not None else None
        self._wh = img_wh
        self._count = -1
        self._rand_shift_range = rand_shift_range
        # Get 1d list of polygons
        polygons = flatten_2dlist([json_record_to_poly(record) for record in self._records_json])
        self._polygons_centroid = [np.round(polygon.centroid) for polygon in polygons]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._records_json)

    def __next__(self):
        self._count += 1
        if self._count < len(self._records_json):
            return self.__getitem__(self._count)
        else:
            self._count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        y,x = self._polygons_centroid[idx]
        w,h = self._wh
        y,x = y-h//2, x-w//2 # align center of crop with poly
        window = ((x, x+w),(y, y+h))
        img = self._img.read(window=window, boundless=True)
        mask = self._mask.read(window=window, boundless=True)
        if self._border is not None:
            return img, mask, self._border.read(window=window, boundless=True)

        return img, mask

    def __del__(self):
        del self._mask
        del self._img


class BackgroundSampler:
    """Generates tuples of img and mask without glomeruli.
    """

    def __init__(self,
                 img_path: str,
                 mask_path: str,
                 polygons: List[geometry.Polygon],
                 img_wh: Tuple[int, int],
                 num_samples: int,
                 step: int = 25,
                 max_trials: int = 25,
                 mask_glom_val: int = 255,
                 buffer_dist: int = 0,
                 border_path=None,
                 strict_mode=True
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
           max_trials: max number of trials per one iteration
           step: num of glomeruli between iterations
           mask_glom_value: mask pixel value containing glomerulus

        Example:
            # Get list of cortex polygons
            polygons = utils.get_cortex_polygons(utils.jread(img_anot_struct_path))
        """

        self._mask = TFReader(mask_path)
        self.mask_path = mask_path
        self._img = TFReader(img_path)
        self._border = rasterio.open(border_path) if border_path is not None else None
        self._polygons = [poly.buffer(buffer_dist) for poly in polygons] if polygons else None # Dilate if any
        self._w, self._h = img_wh
        self._num_samples = num_samples
        self._mask_glom_val = mask_glom_val
        self._boundless = True
        self._count = -1
        self._step = step
        self._max_trials = max_trials
        self._strict_mode = strict_mode

        # Get list of centroids
        self._centroids = [self.gen_backgr_pt() for _ in range(num_samples)]

    def gen_pt_in_img(self):
        W, H = self._img.shape
        pt = np.random.random() * W + self._w, np.random.random() * H + self._h # lazy
        return pt

    def gen_backgr_pt(self) -> Tuple[int, int]:
        """Generates background point.
        Idea is to take only <self._max_trials> trials, if point has not been found, then increment permissible
        num of glomeruli inside background by <self._step>.
        """

        glom_presence_in_backgr, trial = 0, 0

        gen = partial(gen_pt_in_poly, polygon=random.choice(self._polygons), max_num_attempts=200) \
            if self._polygons is not None else self.gen_pt_in_img

        while True:
            rand_pt = gen()
            x_cent, y_cent = np.array(rand_pt).astype(int)
            x_off, y_off = x_cent - self._w // 2, y_cent - self._h // 2
            # Reverse x and y, because gdal return C H W

            window = Window(x_off, y_off, self._w, self._h)
            sample_mask = self._mask.read(window=window, boundless=self._boundless)
            trial += 1 

            if self._strict_mode:
                if np.sum(sample_mask) <= glom_presence_in_backgr * self._mask_glom_val:
                    return x_cent, y_cent
                elif trial == self._max_trials:
                    trial, glom_presence_in_backgr = 0, glom_presence_in_backgr + self._step
            else:
                return x_cent, y_cent


    def __iter__(self):
        return self

    def __len__(self):
        return self._num_samples

    def __next__(self):
        self._count += 1
        if self._count < self._num_samples:
            return self.__getitem__(self._count)
        else:
            self._count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x_off = self._centroids[idx][0] - self._w // 2
        y_off = self._centroids[idx][1] - self._h // 2

        window = Window(x_off, y_off, self._w, self._h)
        img = self._img.read(window=window, boundless=self._boundless)
        mask = self._mask.read(window=window, boundless=self._boundless)
        if self._border is not None:
            return img, mask, self._border.read(window=window, boundless=True)
        return img, mask

    def __del__(self):
        del self._mask
        del self._img



class PolySampler:
    """Generates images from polygon
    """

    def __init__(self,
                 img_path: str,
                 polygons: List[geometry.Polygon],
                 img_wh: Tuple[int, int],
                 num_samples: int,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Example:
            # Get list of cortex polygons
            polygons = utils.get_cortex_polygons(utils.jread(img_anot_struct_path))
        """
        buffer_dist = 0
        self._img = rasterio.open(img_path)
        self._polygons = [poly.buffer(buffer_dist) for poly in polygons] 
        self._w, self._h = img_wh
        self._num_samples = num_samples
        self._boundless = True
        self._count = -1

    def gen_pt(self) -> Tuple[int, int]:
        # TODO refact
        gen = partial(gen_pt_in_poly, random.choice(self._polygons)) 
        rand_pt = gen()
        x_cent, y_cent = np.array(rand_pt).astype(int)
        return x_cent, y_cent

    def __next__(self):
        self._count += 1
        if self._count < self._num_samples:
            return self.__getitem__(self._count)
        else:
            self._count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x_cent, y_cent = self.gen_pt()
        x_off = x_cent - self._w // 2
        y_off = y_cent - self._h // 2

        window= Window(x_off, y_off, self._w, self._h)
        img = self._img.read(window=window, boundless=self._boundless)
        return img

    def __iter__(self): return self
    def __len__(self): return self._num_samples
    def __del__(self): del self._img

class GridSampler:

    def __init__(self,
                 img_path: str,
                 mask_path: str,
                 img_wh: Tuple[int, int],
                 ) -> Tuple[np.ndarray, np.ndarray]:

        self._mask = TFReader(mask_path)
        self._img = TFReader(img_path)
        self._w, self._h = img_wh
        self._boundless = True
        self._count = -1

        _, dims, *_  = get_basics_rasterio(img_path)
        self.block_cds = list(generate_block_coords(dims[0], dims[1], img_wh))
        self._num_samples = len(self.block_cds)

    def __iter__(self): return self
    def __len__(self): return self._num_samples

    def __next__(self):
        self._count += 1
        if self._count < self._num_samples:
            return self.__getitem__(self._count)
        else:
            self._count = -1
            raise StopIteration("Failed to proceed to the next step")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:

        y_off, x_off, _, _ = self.block_cds[idx]
        window = Window(x_off, y_off, self._w, self._h)
        img = self._img.read(window=window, boundless=self._boundless)
        mask = self._mask.read(window=window, boundless=self._boundless)
        return img, mask

    def __del__(self):
        del self._mask
        del self._img


def _write_block(block, name):
    x, y, block_data = block
    #print(name, x,y,block_data.shape, block_data.dtype)
    t = Image.fromarray(block_data.transpose((1,2,0)))
    t.save(f'output/{name}_{x}_{y}.png')

def tif_block_read(name, block_size=None):
    if block_size is None: block_size = (256, 256)
    input_file, (W,H), _ = get_basics_rasterio(name)

    nXBlocks, nYBlocks = _count_blocks(name, block_size=block_size)
    nXValid, nYValid = block_size[0], block_size[1]
    
    for X in range(nXBlocks):
        if X == nXBlocks - 1: nXValid = W - X * block_size[0]
        myX = X * block_size[0]
        nYValid = block_size[1]
        for Y in range(nYBlocks):
            if Y == nYBlocks - 1: nYValid = H - Y * block_size[1]
            myY = Y * block_size[1]
            
            window = Window(myY, myX, nYValid, nXValid)
            block = input_file.read([1,2,3], window=window)
            #print(myX, myY, nXValid, nYValid, W, H, block.shape)

            yield X, Y, block
    del input_file



def _count_blocks(name, block_size=(256, 256)):
    # find total x and y blocks to be read
    _, dims, *_  = get_basics_rasterio(name)
    nXBlocks = (int)((dims[0] + block_size[0] - 1) / block_size[0])
    nYBlocks = (int)((dims[1] + block_size[1] - 1) / block_size[1])
    return nXBlocks, nYBlocks

def generate_block_coords(H, W, block_size):
    h,w = block_size
    nYBlocks = (int)((H + h - 1) / h)
    nXBlocks = (int)((W + w - 1) / w)
    
    for X in range(nXBlocks):
        cx = X * h
        for Y in range(nYBlocks):
            cy = Y * w
            yield cy, cx, h, w

