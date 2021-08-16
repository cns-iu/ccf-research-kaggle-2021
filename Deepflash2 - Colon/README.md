# HuBMAP---Hacking-the-Kidney
This folder is to reproduce the deepflash2 solution for HuBMAP- Hacking the Kidney Challenge on the colon dataset by training it from the scratch or by employing transfer learning. 

Training from scratch:
1. Data preprocessing - Convert the images into zarr format, and calculate the pdfs for the images (use zarr and labels code file)
2. Training - Train on the colon data and save the 5 fold models (use train-scratch code file)
3. Inference - Get the predictions on the test images using the trained models (use the inference code file)
4. Performance Evaluation - Calculate the performance metrics for the predicted data (use the performancemetrics code file)

Training by transfer learning:
Most parts will remain the same. 
1. Data preprocessing - Convert the images into zarr format, and calculate the pdfs for the images (use zarr and labels code file)
2. Training - Train on the colon data by using the models provided by the team as the initial weights and save the 5 fold models (use train code file)
3. Inference - Get the predictions on the test images of colon data using the trained models (use the inference code file, just change the model path)
4. Performance Evaluation - Calculate the performance metrics for the predicted data (use the performancemetrics code file, just change the predicted csvs path)
