import imageio
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, io
import copy
import sys
from skimage.transform import rescale, resize
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# read mask image
image_path = r'C:\Users\yiju\Desktop\Copy\Scripts\masks\1-tom-mask\pred_2ec3f1bb9.png'
resize_index = 1
if len(sys.argv) >= 2:
    image_path = sys.argv[1]
if len(sys.argv) >= 3:
    resize_index = int(sys.argv[2])
preview = False

img = imageio.imread(image_path)
# img = io.imread(image_path)

if len(img.shape) == 2:
    mask = img
else:
    mask = np.array(img[:, :, 0])
mask = np.where(mask > 127, 1, 0)
# mask = resize(mask, (500,500), anti_aliasing=False)

# get the contour
contours = measure.find_contours(mask, 0.8)

# contour to polygon
polygons = []
for object in contours:
    coords = []
    for point in object:
        coords.append([int(point[0]), int(point[1])])
    polygons.append(coords)
if preview:
    print(polygons)

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
json_path = image_path.replace('png', 'json')
with open(json_path, 'w') as fp:
    json.dump(geojson_list, fp, indent=2)

# polygon preview
if preview:
    new_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for polygon in polygons:
        for point in polygon:
            new_img[int(point[0]), int(point[1])] = 255
    plt.imshow(new_img)
    # plt.imsave(r'x:\mask_88.png', new_img)
    plt.show()
