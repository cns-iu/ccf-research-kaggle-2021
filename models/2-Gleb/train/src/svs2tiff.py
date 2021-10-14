import numpy as np
import rasterio
from pathlib import Path
import argparse


def svs2tiff(path_in, path_out):
    svs_files = list((Path(path_in)).glob('*.svs'))
    for svs_file in svs_files:
        print(f"{svs_file}")
        img = rasterio.open(f"{svs_file}", 'r')
        img_array = img.read()
        h = img_array.shape[1]
        w = img_array.shape[2]
        print(w, h)
        svs_name = svs_file.with_suffix('').name
        img_tiff = rasterio.open(f"{path_out}/{svs_name}.tiff", 'w', driver='GTiff', height=h, width=w, count=3, nbits=8, dtype=np.uint8)
        img_tiff.write(img_array)
        img_tiff.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVS path')
    parser.add_argument('path_in', type=str, help='SVS path')
    parser.add_argument('path_out', type=str, help='TIFF path')
    args = parser.parse_args()
    svs2tiff(args.path_in, args.path_out)
    print(args.path)


'''
if __name__ == '__main__':
    path_in = 'F:/kaggle/dsmatch/k7nvtgn2x6-3/DATASET_A_DIB'
    path_out = 'F:/kaggle/dsmatch/k7nvtgn2x6-3/DATASET_A_DIB/Tiff'
    svs2tiff(path_in, path_out)
'''

