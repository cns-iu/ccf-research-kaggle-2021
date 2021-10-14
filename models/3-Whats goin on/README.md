# 3-Whats goin on

### Description
- Model training uses an ensemble of 2 sets of 5-fold models using the U-Net architecture (pretrained on ImageNet) with resnet50_32x4d and resnet101_32x4d64 as backbones, respectively. 
- Additionally, a Feature Pyramid Network (FPN) is added to provide skip connections between upscaling blocks of the decoder, atrous spatial pyramid pooling (ASPP) is added to enlarge receptive fields, and pixel shuffle is added instead of transposed convolution to avoid artifacts. 
- The model reads kidney/colon data downsampled by a factor of 2 and tiles of size 1024x1024 are sampled and filtered based on a saturation threshold of 40. 
- The models are trained for 50 epochs each, using a one cycle learning rate scheduler with pct_start=0.2, div_factor=1e2, max_lr=1e-4, batch size of 16. 
- The model uses an expansion tile size of 32. The model uses binary cross entropy loss, with gradient norm clipping at 1 and Adam optimizer.
- For the model trained on data from scratch or using transfer learning, the batch size is increased to 64 and the expansion tile size is increased to 64.

### Requirements
- Python 3
- PyTorch
- Pandas
- Numpy
- CUDA
- See `requirements.txt` file for detailed list of requirements.

### Usage
- Run the `Makefile` to set up the environment and install all the dependencies.
- Run `install_additional_packages.sh` to install additional dependencies.
- Use `reproduce.sh` file for training.
- Use `WhatsGoinOn-Inference-Kidney.ipynb` notebook for inference on the kidney data.
- Use `WhatsGoinOn-Inference-Colon.ipynb` notebook for inference on the colon data.
- Use `/utils/Performance Metrics - Kidney.ipynb` to calculate the performance metrics for the model on kidney data.
- Use `/utils/Performance Metrics - Colon.ipynb` to calculate the performance metrics for the model on colon data.
