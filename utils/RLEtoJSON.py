import numpy as np
from skimage import measure
import json
import copy
import matplotlib.pyplot as plt

def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        enc_split = enc.split()
        for i in range(len(enc_split) // 2):
            start = int(enc_split[2 * i]) - 1
            length = int(enc_split[2 * i + 1])
            img[start: start + length] = 1 + m
    print (len(img), img)
    print ("Is mask empty?", np.count_nonzero(img))
    return img.reshape(shape).T


def mask2json(mask):
    contours = measure.find_contours(mask, 0.8)

    # contour to polygon
    polygons = []
    for object in contours:
        coords = []
        for point in object:
            coords.append([int(point[0]), int(point[1])])
        polygons.append(coords)

    # save as json
    geojson_dict_template = {
        "type": "Feature",
        "id": "PathAnnotationObject",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
            ]
        },
        "properties": {
            "classification": {
                "name": "glomerulus",
                "colorRGB": -3140401
            },
            "isLocked": True,
            "measurements": []
        }
    }
    geojson_list = []
    for polygon in polygons:
        geojson_dict = copy.deepcopy(geojson_dict_template)
        geojson_dict["geometry"]["coordinates"].append(polygon)
        geojson_list.append(geojson_dict)

    return geojson_list



import pandas as pd
import rasterio

header_list = ["id","predicted"]
df_pred = pd.read_csv('submission_kidney_new_1.csv')
print (df_pred.columns)
IMG_PATH = '/N/slate/soodn/new-kaggle-data/Vanderbilt Kidney Dataset/'
for id, encs in df_pred.iterrows():
    print (encs.id)
    img = rasterio.open(IMG_PATH + str(encs.id) + ".tiff")
    print ("*****",img.shape)
    mask = enc2mask(encs.predicted, img.shape)
    plt.imsave("mask.png", mask)
    print ("Mask made.")
    geojson_list = mask2json(mask)
    print ("Json made.")
    json_path = IMG_PATH + str(encs.id) + '.json'
    with open(json_path, 'w') as fp:
        json.dump(geojson_list, fp, indent=2)
    print ("Json saved.")
    
