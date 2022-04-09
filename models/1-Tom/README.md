# 1-Tom

### Description
- The model uses a single U-Net SeResNext101 architecture with Convolutional Block Attention Module (CBAM), hypercolumns, and deep supervision. 
- It reads the WSIs as tiled 1024x1024 pixel images and then further resized as 320x320 tiles and sampled using a balanced sampling strategy. 
- The model is trained using a combination of Binary Cross-entropy loss and Lovász Hinge loss, and the optimizer used is SGD (Stochastic gradient descent). Training is for 20 epochs, with a learning rate of 1e-4 to 1e-6 and batch size of 8.
- For the model trained on colon data from scratch or using transfer learning, the training is done for 50-100 epochs and the validation set is increased from 1 slide to 2 slides.

### Requirements
- albumentations
- opencv-python
- pandas
- torch
- tqdm
- Python 3
- CUDA 
- cuddn 
- nvidia drivers 
- See `kaggle-hubmap-main/requirements.txt` file for a detailed list of dependencies.

### Usage
- Use Inference.py to run inference on a dataset.
- Use train folder to train on a dataset.
    cd src
    cd 01_data_preparation/01_01
    python data_preparation_01_01.py
    cd ..
    cd 01_02
    python data_preparation_01_01.py
    cd ..
    cd ..
    cd 02_train
    python train_02.py

