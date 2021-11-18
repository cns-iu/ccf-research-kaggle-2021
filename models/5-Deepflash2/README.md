# 5-Deepflash2

### Description
- The model architecture used is a simple U-Net architecture with an efficientnet-b2 encoder (pretrained on ImageNet). 
- Input data is converted and stored as .zarr file format for efficient loading on runtime. 
- The model collectively employs two sampling approaches: 1) Sampling tiles that contain all glomeruli (to ensure that each glomerulus is seen at least once during each epoch of training). 2) Sampling random tiles based on region (cortex, medulla, background) probabilities (to give more weight to the cortex region during training since glomeruli are mainly found in the cortex). 
- The region sampling probabilities were chosen based on expert knowledge and experiments: 0.7 for cortex, 0.2 for medulla, and 0.1 for background. 
- On runtime, the model samples tiles of size 512x512 and uses a resolution downscale factor of 2, 3, and 4 in subsequent runs. 
- Model training uses Dice-Crossentropy loss, [Ranger optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) (a combination of RAdam and LookAhead optimizer), a maximum learning rate of 1e-3, and a batch size of 16. 
- The models are trained and tested using 5-fold cross validation.
- For the model trained on the colon data (both with and without transfer learning), the background probability is set to 0.1 and the colon probability is set to 0.9 for sampling, since the colon data lacks the masks for anatomical structures. 
- A weight decay of 1e-5 was added (for the model trained without transfer learning). 
- For the transfer learning model, saved weights are loaded from the model trained on kidney data at 3x downsampling and the first 13 parameter groups are frozen during training. 

### Requirements
- Python 3
- PyTorch
- Pandas
- Numpy
- CUDA
- fastai
- albumentations
- rasterio

### Usage
- Use `DeepFlash2-Zarr-Colon.ipynb` to convert all data to `.zarr` format.
- Use `DeepFlash2-Labels-Colon.ipynb` to generate labels.
- Use `DeepFlash2-Train-Colon.ipynb` notebook for training. (If you encounter odd memory-related errors while importing the packages, try exporting the notebook as a .py file and then run using `python DeepFlash2-Train-Colon.ipynb`)
- Use `DeepFlash2-Inference-Kidney.ipynb` notebook for inference on the kidney data.
- Use `Deepflash-inference-colon.py` file for inference on the colon data.
- Use `/utils/Performance Metrics - Kidney.ipynb` to calculate the performance metrics for the model on kidney data.
- Use `/utils/Performance Metrics - Colon.ipynb` to calculate the performance metrics for the model on colon data.
