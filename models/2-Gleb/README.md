# 2-Gleb

### Description
- The model is trained using an ensemble of four 4-fold models namely, Unet-regnety16, Unet-regnetx32, UnetPlusPlus-regnety16, and Unet-regnety16 with scse attention decoder.
-  The model reads tiles of size 1024x1024 sampled from the kidney/colon data. 
-  The models are trained for 50-80 epochs each, with a learning rate of 1e-4 to 1e-6, and batch size of 8. 
-  The loss function is Dice coefficient loss and the optimizer used is AdamW.
-  For the model trained on data from scratch or using transfer learning, the model is trained for 50-100 epochs and the sampling downscale factor is changed from 3 to 2. 

### Requirements
- Python 3
- PyTorch
- Pandas
- Numpy
- CUDA
- See `train/requirements.txt` and `train/requirements_def.txt` files for detailed list of requirements.

### Usage
- Use `SecondWinnerGleb_1_kidney.ipynb` notebook for training and inference on the kidney data.
- Use `SecondWinnerGleb_4_colon.ipynb` notebook for training and inference on the colon data.
- Use `/utils/Performance Metrics - Kidney.ipynb` to calculate the performance metrics for the model on kidney data.
- Use `/utils/Performance Metrics - Colon.ipynb` to calculate the performance metrics for the model on colon data.
