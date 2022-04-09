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
- Save the data in the input folder.
- Use Inference.py to run inference on a data.
- Use train folder to train on a dataset as follows: 
    cd train
    python src/main.py --cfg src/configs/unet_model_0.yaml
    python src/main.py --cfg src/configs/unet_model_1.yaml
    python src/main.py --cfg src/configs/unet_model_2.yaml
    python src/main.py --cfg src/configs/unet_model_3.yaml