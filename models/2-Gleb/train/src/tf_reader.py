from typing import Tuple
import numpy as np
import rasterio
from rasterio.windows import Window


class TFReader:
    """Reads tiff files.

    If subdatasets are available, then use them, otherwise just handle as usual.
    """

    def __init__(self, path_to_tiff_file: str):
        self.ds = rasterio.open(path_to_tiff_file)
        self.subdatasets = self.ds.subdatasets
        self.is_subsets_avail = len(self.subdatasets) > 0
        if self.is_subsets_avail:
            path_to_subdatasets = self.ds.subdatasets
            self.list_ds = [rasterio.open(path_to_subdataset)
                            for path_to_subdataset in path_to_subdatasets]

    def read(self, window: Tuple[None, Window] = None, boundless: bool=True):
        if self.is_subsets_avail:
            output = np.vstack([ds.read() for ds in self.list_ds]) if window is None else \
                np.vstack([ds.read(window=window, boundless=boundless) for ds in self.list_ds])
        else:
            output = self.ds.read() if window is None else \
                self.ds.read(window=window, boundless=boundless)
        return output

    @property
    def shape(self):
        return self.ds.shape

    def __del__(self):
        del self.ds
        if self.is_subsets_avail:
            del self.list_ds
