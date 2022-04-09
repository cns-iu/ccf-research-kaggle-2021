# **About** : This notebook is used to downscale images in the train and test set, in order to speed-up training and inference
#   - Use the `FACTOR` parameter to specify the downscaling factor. We recommend generating data of downscaling 2 and 4.
#   - For training data, we save extra time by also computing downscaling rles. Use the `NAME` parameter to specify which rle to downscale.
#   - It is only require to save the downscaled images once, use the `SAVE_IMG` parameters to this extent.

import os
import gc
import cv2
import sys
import tifffile
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from collections import Counter
from matplotlib import pyplot as plt

sys.path.append("../code")

import os
import time
start = time.time()

from data.dataset import load_image
from utils.rle import *
from params import *

def elapsed_time(start_time):
    return time.time() - start_time

FACTOR = 2

### Train
out_dir = DATA_PATH + f"train_{FACTOR}/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

SAVE_IMG = True

names = ["_onlyfc", "_fix", ""] # unhealthy class, healthy class with fixed issues, original data
for NAME in names:
    if dataset == "kidney":
        df_masks = pd.read_csv(DATA_PATH + "train" + NAME + ".csv").set_index("id")
    elif dataset == "colon":
        df_masks = pd.read_csv(DATA_PATH + "/train" + NAME + ".csv")
        df_masks = df_masks[df_masks.id != 'HandE_B005_CL_b_RGB_topright']
        df_masks.index = df_masks.id

    masks = {}

    for index, encs in tqdm(df_masks.iterrows(), total=len(df_masks)):
        # read image and generate the mask
        if index == "HandE_B005_CL_b_RGB_topright":
            continue
        img = load_image(os.path.join(TIFF_PATH, index + ".tiff"))
        mask = enc2mask(encs, (img.shape[1], img.shape[0]))

        if SAVE_IMG:
            img = cv2.resize(
                img,
                (img.shape[1] // FACTOR, img.shape[0] // FACTOR),
                interpolation=cv2.INTER_AREA,
            )
            tifffile.imsave(out_dir + f"{index}.tiff", img)

        mask = cv2.resize(
            mask,
            (mask.shape[1] // FACTOR, mask.shape[0] // FACTOR),
            interpolation=cv2.INTER_NEAREST,
        )
        
        rle = mask2enc(mask)
        masks[index] = rle
        
    #     break

    df_masks = pd.DataFrame.from_dict(masks).T.reset_index().rename(columns={0: "encoding", "index": "id"})
    df_masks.to_csv(f"{DATA_PATH}train_{FACTOR}{NAME}.csv", index=False)
    print(f"Saved data to {DATA_PATH}train_{FACTOR}{NAME}.csv")

out_dir = DATA_PATH + f"test_{FACTOR}/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

df = pd.read_csv(DATA_PATH + "sample_submission.csv")

for index in tqdm(df['id']):
    # read image and generate the mask
    img = load_image(os.path.join(TIFF_PATH_TEST, index + ".tiff"))
    img = cv2.resize(
        img,
        (img.shape[1] // FACTOR, img.shape[0] // FACTOR),
        interpolation=cv2.INTER_AREA,
    )
    
    tifffile.imsave(out_dir + f"{index}.tiff", img)

# EXTRA_IMGS
# for index in tqdm(EXTRA_IMGS):
#     print (index)
#     # read image and generate the mask
#     img = load_image(os.path.join(TIFF_PATH_TEST, index + ".tiff"))

#     img = cv2.resize(
#         img,
#         (img.shape[1] // FACTOR, img.shape[0] // FACTOR),
#         interpolation=cv2.INTER_AREA,
#     )
    
#     tifffile.imsave(out_dir + f"{index}.tiff", img)

print ("Run time = ", elapsed_time(start))
