# **About** : This notebook is used to retrieve hand-made annotations. 
#   - Use the `ADD_FC` and `ONLY_FC` parameters to generate labels for the healthy and unhealthy classes.
#   - Use the `SAVE_TIFF `parameter to save the external data as tiff files of half resolution.
#   - Use the `PLOT` parameter to visualize the masks.
#   - Use the `SAVE` parameter to save the masks as rle. # %% [markdown]

import os
import sys
import cv2
import json
import glob
import rasterio
import tifffile
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

sys.path.append("../code/")

import time
start = time.time()

from params import *
from utils.rle import *
from data.dataset import load_image
from utils.plots import plot_contours_preds

def elapsed_time(start_time):
    return time.time() - start_time

IDENTITY = rasterio.Affine(1, 0, 0, 0, 1, 0)

df_info = pd.read_csv(DATA_PATH + f"HuBMAP-20-dataset_information.csv")
df_mask = pd.read_csv(DATA_PATH + "train.csv")

ANNOT_PATH = DATA_PATH + "annot/"

if dataset == "colon":
    df_mask = df_mask.rename(columns={"predicted":"encoding"})
    df_mask = df_mask[df_mask.id != 'HandE_B005_CL_b_RGB_topright']

PLOT = False
add_fc_list = [True, False]
only_fc_list = [True, False]

for ADD_FC in add_fc_list:
    for ONLY_FC in only_fc_list:
        new_df = df_mask.copy().set_index('id')
        if ONLY_FC:
            new_df['encoding'] = ""

        for id_ in tqdm(df_mask['id']):
            print(f' -> {id_}')
            if id_ + ".json" in os.listdir(ANNOT_PATH):        
                annot = json.load(open(ANNOT_PATH + id_ + ".json", 'r'))
                
                w, h = df_info[df_info['image_file'] == id_][['width_pixels', 'height_pixels']].values[0]
                
                rle = df_mask[df_mask['id'] == id_]['encoding']
                
        #       mask = enc2mask(rle, (w, h)).astype(np.uint8)  # smh not working
                mask = np.zeros((h, w), dtype=np.uint8)
                if not ONLY_FC:
                    mask += enc2mask(rle, (w, h)).astype(np.uint8)
                
                added = 0
                for info in annot:
                    label = info['properties']['classification']['name']

                    if (not ADD_FC) and (label == "FC"):
                        continue
                            
                    if ONLY_FC and label != "FC":
                        continue

                    poly = info['geometry']['coordinates']
                    try:
                        mask = cv2.fillPoly(mask, np.int32([poly]), True)
                    except ValueError:
                        poly = np.concatenate([np.array(poly[i]).squeeze() for i in range(len(poly))])
                        mask = cv2.fillPoly(mask, np.int32([poly]), True)
                    added +=1
                    
                print(f"Added {added} glomerulis")
                
                new_df.loc[id_] = rle_encode_less_memory(mask)
                
                if PLOT:
                    img = load_image(os.path.join(TIFF_PATH, id_ + ".tiff"), full_size=False)
                    
                    mask = cv2.resize(
                        mask,
                        (w // 4, h // 4),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    assert mask.shape == img.shape[:2], (mask.shape, img.shape)
                
                    fig = plot_contours_preds(img, mask, w=1, downsize=4)
                    w = 1000
                    h = int(w *  mask.shape[0] / mask.shape[1])
                    fig.update_layout(
                        autosize=False,
                        width=w,
                        height=h,
                    )

                    fig.show()

                    break

        if not PLOT:
            name = "train_fix.csv" if not ADD_FC else "train_fc.csv"
            if ONLY_FC:
                name = "train_onlyfc.csv"
            new_df.to_csv(DATA_PATH + name)
            print (name)
            print(f'\n -> Saved masks to {DATA_PATH + name}')

PLOT = False
SAVE_TIFF = True
SAVE = True
add_fc_list = [True, False]
only_fc_list = [True, False]

files = [p for p in os.listdir(DATA_PATH + "extra/") if p.endswith("svs")]
rles = {}

for ADD_FC in add_fc_list:
    for ONLY_FC in only_fc_list:
        for file in tqdm(files):
            id_ = file[:-4]
            print(f' -> {id_}')

        #     if id_ != "SAS_21908_001":
        #         continue

            if os.path.exists(ANNOT_PATH + id_ + ".json"):
                original_img = rasterio.open(DATA_PATH + "extra/" + file, transform=IDENTITY, num_threads='all_cpus')
                img = original_img.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)

                shape = img.shape[:2]
            
                annot = json.load(open(ANNOT_PATH + id_ + ".json", 'r'))

                mask = np.zeros(shape, dtype=np.uint8)

                added = 0
                for info in annot:
                    poly = np.array(info['geometry']['coordinates'])
                
                    try:
                        label = info['properties']['classification']['name']
                    except KeyError:
                        label = "G"
                
                    if not ADD_FC and label == "FC":
                        continue

                    if ONLY_FC and label != "FC":
                        continue
                    
                    poly = info['geometry']['coordinates']
                    try:
                        mask = cv2.fillPoly(mask, np.int32([poly]), True)
                    except ValueError:
                        poly = np.concatenate([np.array(poly[i]).squeeze() for i in range(len(poly))])
                        mask = cv2.fillPoly(mask, np.int32([poly]), True)
                    added += 1
            
                print(f"Added {added} glomerulis")
            
                if PLOT:
                    print('plot')
                    fig = plot_contours_preds(img, mask, w=2, downsize=8)

                    w = 1000
                    h = int(w *  mask.shape[0] / mask.shape[1])
                    fig.update_layout(
                        autosize=False,
                        width=w,
                        height=h,
                    )

                    fig.show()

                    break
                
                if SAVE:
                    if SAVE_TIFF:
                        img = cv2.resize(
                            img,
                            (img.shape[1] // 2, img.shape[0] // 2),
                            interpolation=cv2.INTER_AREA,
                        )
                        
                        if not os.path.exists(DATA_PATH + "extra_tiff/"):
                            os.mkdir(DATA_PATH + "extra_tiff/")
                        tifffile.imsave(DATA_PATH + "extra_tiff/" + f"{id_}.tiff", img)

                    mask = cv2.resize(
                        mask,
                        (mask.shape[1] // 2, mask.shape[0] // 2),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    print (id_, mask)
                    rles[id_] = rle_encode_less_memory(mask)

        df_annot_extra = pd.DataFrame.from_dict(rles, orient='index', columns=['encoding'])
        df_annot_extra['id'] = df_annot_extra.index
        df_annot_extra.reset_index(drop = True)

        if SAVE and not PLOT:
            name = "train_extra.csv" if not ADD_FC else "train_extra_fc.csv"
            if ONLY_FC:
                name = "train_extra_onlyfc.csv"
            df_annot_extra.to_csv(DATA_PATH + name)
            print(f'\n -> Saved masks to {DATA_PATH + name}')

print ("Run time = ", elapsed_time(start))