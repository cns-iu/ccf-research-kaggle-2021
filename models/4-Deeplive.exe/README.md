# 4-Deeplive.exe

### Description
- The model architecture used is a simple U-net with an efficientnet-b1 encoder. 
- The model employs a dynamic sampling approach whereby it samples tiles of size 512x512 (at a resolution downscale factor of 2) and 768x768 (at a resolution downscale factor of 3). 
- The tiles are sampled from regions having visible glomeruli in them based on annotations, instead of sampling randomly. 
- Model training uses the cross-entropy loss, Adam optimizer, an adaptive learning rate (linearly increased up to 0.001 during the first 500 iterations and then linearly decreased to 0), and a batch size of 32. 
- The model is trained using 5-fold cross validation for at least 10,000 iterations.
- For the model trained on colon data from scratch, an overlap factor of 1 is used. 
- For the model trained on colon data using transfer learning, on_spot_sampling is set to 1 and an overlap factor of 1 is used.

### Requirements
- Python 3
- PyTorch
- Pandas
- Numpy
- CUDA

### Usage
- Use `Training.ipynb` file for training.
- Use `Deeplive-Inference-Kidney.ipynb` notebook for inference on the kidney data.
- Use `Deeplive-Inference-Colon.ipynb` notebook for inference on the colon data.
- Use `/utils/Performance Metrics - Kidney.ipynb` to calculate the performance metrics for the model on kidney data.
- Use `/utils/Performance Metrics - Colon.ipynb` to calculate the performance metrics for the model on colon data.
