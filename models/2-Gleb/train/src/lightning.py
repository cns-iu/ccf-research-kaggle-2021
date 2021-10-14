import numpy as np
import cv2
from typing import Tuple
from collections import Counter

import utils


def cut_off_light_from_img(img: np.ndarray,
                           threshold: int = 210,
                           kernel_size: int = 5,
                           min_light_ratio: float = 0.005,
                           max_light_ratio: float = 0.5,
                           max_gray_val: int = 255,
                           is_morph_before_thresh: bool = True,
                           is_morph_after_thresh: bool = True) -> Tuple[None, np.ndarray]:
    """Cuts off lightning from img.

    Gets np.ndarray (3, h, w) as input and returns np.ndarray (h, w) or None.

    ratio: Defines the ratio of kernel size compared to img.
           Smaller value, thinner lightnings (varies from 0.0 to 1.0)
    min_light_cell_ratio: Min possible ratio (total number of white cells / total number of cells)
    max_light_ratio: Max possible ratio (total number of white cells / total number of cells)

    Attention: Img may contain only 3 channels.

    """

    num_channels, h, w = img.shape
    assert num_channels == 3, "Invalid number of channels"

    if is_morph_before_thresh or is_morph_after_thresh:
        kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))
        small_kernel = cv2.getStructuringElement(1, (kernel_size // 2, kernel_size // 2))

    img = utils.rgb2gray(img).astype(np.uint8)

    if is_morph_before_thresh:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    _, thresh = cv2.threshold(img, threshold, max_gray_val, cv2.THRESH_BINARY)

    if is_morph_after_thresh:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    total_num_cells = w * h
    num_white_cells = thresh.sum() / max_gray_val
    ratio = num_white_cells / total_num_cells

    if min_light_ratio <= ratio <= max_light_ratio:
        return thresh.astype(np.uint8)  # (h, w)
