import cv2, zarr, tifffile
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from pathlib import Path

import time
start = time.time()

def elapsed_time(start_time):
    return time.time() - start_time

def read_image(image_id, path, scale=None, verbose=1):
    "Load images with ID from path" 
    try: 
        image = tifffile.imread(path+f"/train/{image_id}.tiff")
    except:
        image = tifffile.imread(path+f"/test/{image_id}.tiff")
    
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    elif image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    if verbose:
        print(f"[{image_id}] Image shape: {image.shape}")
    
    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)
        
        if verbose:
            print(f"[{image_id}] Resized Image shape: {image.shape}")  
    return image

# <h6> Step 2 - Set paths for the files needed </h6>
scale = 2
BASE_PATH = r'/N/slate/soodn/'
dataset = "colon" 
# dataset = "kidney" 

INPUT_PATH = BASE_PATH+'hubmap-'+dataset+'-segmentation'

df_train = pd.read_csv(INPUT_PATH+"/train.csv")
df_sample = pd.read_csv(INPUT_PATH+"/sample_submission.csv")
g_out = zarr.group(f'output_{dataset}/images_scale{scale}')

if dataset == "colon":
    df_sample = df_sample.rename(columns={"predicted":"encoding"})
    df_train = df_train[df_train.id != 'HandE_B005_CL_b_RGB_topright']

# <h6> Step 3 - Rescale the images, and save the rescaled images to a folder </h6>
for idx in df_sample['id'].tolist()+df_train['id'].tolist():
    print("idx", idx)
    img = read_image(idx, INPUT_PATH, scale=scale)
    g_out[idx] = img
    print(g_out[idx].info)
    shape = g_out[idx].shape
    
    plt.imshow(cv2.resize(img, dsize=(512, 512*shape[0]//shape[1])))
    plt.show()

print ("Run time = ", elapsed_time(start))
