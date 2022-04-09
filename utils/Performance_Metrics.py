# <h1> HubMap - Hacking the Kidney </h1>
# <h3> Goal - Mapping the human body at function tissue unit level - detect glomeruli FTUs in kidney </h3>
# 
# Description - Calculate the performance metrics for test data predictions of kidney data. <br>
# Input - submission.csv (csv file containing rle format predicted mask), test.csv (csv file containing rle format original mask).<br>
# Output - Performance metrics values - dice coeff, Jaccard index, pixel accuracy, hausdorff distance. <br>
# 
# <b>How to use?</b><br> 
# Change the basepath to where your data lives and you're good to go. <br>
# 
# <b>How to reproduce on a different dataset?</b><br>
# Create a train and test folders of the dataset containing train images and masks and test images and masks respectively. Have a train.csv with the rle for train images and a sample-submission file with test image names. Create a test.csv with rle for test images and predicted csv from the trained network. 
# 
# <hr>
# 
# 
# <h6> Step 1 - Import useful libraries</h6>
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff

# <h6> Step 2 - Write utility functions </h6> 
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
    return img.reshape(shape).T

def dice_scores_img(pred, truth, eps=1e-8):
    pred = pred.reshape(-1) > 0
    truth = truth.reshape(-1) > 0
    intersect = (pred & truth).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)
    dice = (2.0 * intersect + eps) / (union + eps)
    return dice

def perf_metrics(gt, pred):
    n = 0
    d = 0
    for i in range(gt.shape[0]):
        for j in range (gt.shape[1]):
            if (gt[i][j]==pred[i][j]):
                n = n+1
            d = d+1
    return n/d, jaccard_score(gt.flatten(order='C'), pred.flatten(order='C')), directed_hausdorff(gt, pred)

# ##### Step 3 - Calculate mean metrics values for test images 
BASE_PATH = r'/N/slate/soodn/'
dataset = "colon" 
# dataset = "kidney" 
# dataset = "new-data"
INPUT_PATH = BASE_PATH+'hubmap-'+dataset+'-segmentation'

df_pred = pd.read_csv('output/submission_kidney_pvt_deeplive.csv')
df_truth = pd.read_csv(INPUT_PATH+'/test.csv')
df_info = pd.read_csv(INPUT_PATH+'/HuBMAP-20-dataset_information.csv')

scores = []
pa_list = []
ji_list = []
haus_dis_list = []

pvt_test = ['00a67c839', '0749c6ccc', '1eb18739d', '5274ef79a', '5d8b53a68', '9e81e2693', 'a14e495cf', 'bacb03928', 'e464d2f6c',
'ff339c0b2']
for img in pvt_test:
    shape =  df_info[df_info.image_file == img][['width_pixels', 'height_pixels']].values.astype(int)[0]
    truth = df_truth[df_truth['id'] == img]['expected']
    mask_truth = enc2mask(truth, shape)
    pred = df_pred[df_pred['id'] == img]['predicted']
    mask_pred = enc2mask(pred, shape)  
    score = dice_scores_img(mask_pred, mask_truth)
#     pa, ji, haus = perf_metrics(mask_pred, mask_truth)    
#     pa_list.append (pa)
#     ji_list.append(ji)
#     haus_dis_list.append(haus[0])
    scores.append(score)

l = len(df)
for img, s in zip(rles[5:]['id'],scores):
    print (round(s, 3))
        
print ("Average Dice Score = ", round(sum(scores)/l,3))


