<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/cns-iu/ccf-research-ftu">
    <img src="images/cns-logo-1.png" alt="Logo">
  </a>

  <h3 align="center">Common Coordinate Framework (CCF) Research on Functional Tissue Units (FTU)</h3>

  <p align="center">
    FTU Segmentation through Machine Learning (ML) Algorithms proposed by Kaggle team Deepflash2
    <br />
    <a href="https://github.com/cns-iu/ccf-research-ftu"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Solution](#about-the-project)
* [Data](#data)
* [Libraries](#libraries)
* [Methodology](#methodology)
* [How To Use] (#how-to-use)




<!-- ABOUT THE PROJECT -->
## About the Solution

This folder reproduces the kaggle team Deepflash2's solution for HuBMAP- Hacking the Kidney Challenge on the same dataset (kidney); and also tests it on a different dataset (colon). The goal of the challenge is to detect functional tissue units (FTUs) in the kidney tissue using machine learning algorithms. Moreover, how well a model developed for kidney FTU(s), glomeruli, can generalize on colon FTU(s), crypts.

Kaggle competition - https://www.kaggle.com/c/hubmap-kidney-segmentation

## Data
#### Kidney 
The kidney dataset consists 15 high resolution train images and corresponding masks and 5 test images and corresponding masks. It also contains a metadata file containing patients' height weight age gender etc. The image masks are present in two formats - json file or run-length encoding, and can be obtained from one another.

#### Colon 
The colon dataset consists 5 train images and corresponding masks and 2 test images and the corresponding masks. A meta data file, similar to the kidney data, is present as well. 


## Libraries
The code ran on a windows based system with 256GB RAM, i9-10900X Intel Core CPU @ 3.70 GHz.
The specific library versions used are as follows: 
1. cv2 4.4.0
2. zarr 2.8.3
3. tifffile 2020.10.1
4. matplotlib 3.3.4
5. numpy 1.20.3
6. pandas 1.2.3
7. pathlib
8. gc
9. pytorch 1.9.0
10. random
11. segmentation models pytorch
12. fastai 2.5.1
13. deepflash2 0.1.2
14. scipy 1.6.2
15. sklearn 0.24.1
16. augmedical
17. tqdm 4.58.0
18. albumentations 1.0.3
19. wandb 0.11.2
20. sys
21. rasterio 1.1.1
22. glob
23. warnings
24. json 2.0.9
25. PIL 8.1.1

## Methodology
#### Data Pre-processing
1. Augmentation/Preprocessing
2. Efficient Sampling - no "pre-tiling" needed, the training data gets converted into .zarr files for efficient loading.
3. flexible tile dimensions (e.g. 1024, 512, 256) & downscaling (2, 3, 4x) at runtime
4. training focuses on the relevant regions (e.g., tiles that contain glomeruli and cortex)
5. during data augmentation we have no cropping artifacts during rotation etc.
6. Instead of preprocessing the images by saving them into fixed tiles, a combination of two sampling approaches:
    6.1 Sampling tiles via center points in the proximity of every glomerulus - this ensures that each glomerulus is seen during one epoch of training at least once.
    6.2 Sampling random tiles based on region probabilities (e.g., medulla, cortex, other)
We use the provided anatomical information to train with more examples of the cortex than the medulla, because glomeruli have a higher abundance in this region. We also sampled a few examples not contained in the anatomic regions to ensure that our model can interpret these as well.
7. A beneficial side effect of our sampling method is that functional tissue units which were overlooked in the annotation will be rarely used in training.

#### Model
1. Uncertainty Estimation - provides bayesian and energy based measures for uncertainty
2. Trained and tested different encoders (resnet, efficientnetb0-b4) and architectures (U-Net, U-Net-plusplus, deeplab-v3) and found no significant difference in performance. Therefore, decided to use a reasonable small encoder (efficientnet-b2) as well as a simple default U-Net.
3. Final training setup:
Architecture: U-Net
Encoder: efficientnet-b2
Pretraining: imagenet
Loss: Dice-CrossEntropy
Optimizer: ranger
Learning rate: 1e-3
Batch Size: 16
Tile Size: 512x512
Resolultion Downscaling Factor: 3
Training iterations: 2500-3000 (best model selected on validation set)
Ensembling 1: 5 Models (5-fold cross validation)
Ensembling 2: 3 Models at scale 2,3, and 4 trained on all data

4. Loss calculation
Tried the same experiment for different optimizers (SGD, AdamW, Ranger, Madgrad) and found that ranger performed most consistently. In our experiments, the Dice-CrossEntropy loss worked well. We also tried more exotic loss functions like deep supervision but did not find any consistent benefit.

#### Validation
To evaluate the performance of the models, trained and tested them in a five-fold cross validation on kidney data and colon data.


## How to Use
#### To reproduce the solution:
1. Data preprocessing - Convert the images into zarr format, and calculate the pdfs for the images (use zarr and labels code file)
2. Training - Train on the kidney data and save the 5 fold models (use train code file)
3. Inference - Get the predictions on the test images using the trained models (use the inference-kidney code file)
4. Performance Evaluation - Calculate the performance metrics for the predicted data (use the performancemetrics-kidney code file)

#### To generalize on colon data:
Most parts will remain the same. 
1. Data preprocessing - Convert the images into zarr format, and calculate the pdfs for the images (use zarr and labels code file)
2. Training - Train on the kidney data and save the 5 fold models (use train code file)
3. Inference - Get the predictions on the test images of colon data using the trained models (use the inference-colon code file)
4. Performance Evaluation - Calculate the performance metrics for the predicted data (use the performancemetrics-colon code file)
