# HuBMAP---Hacking-the-Kidney
This folder is to reproduce the deepflash2 solution for HuBMAP- Hacking the Kidney Challenge on the same dataset (kidney) as well as generalize on a different dataset (colon). 

To reproduce the solution:
1. Data preprocessing - Convert the images into zarr format, and calculate the pdfs for the images (use zarr and labels code file)
2. Training - Train on the kidney data and save the 5 fold models (use train code file)
3. Inference - Get the predictions on the test images using the trained models (use the inference-kidney code file)
4. Performance Evaluation - Calculate the performance metrics for the predicted data (use the performancemetrics-kidney code file)

To generalize on colon data:
Most parts will remain the same. 
1. Data preprocessing - Convert the images into zarr format, and calculate the pdfs for the images (use zarr and labels code file)
2. Training - Train on the kidney data and save the 5 fold models (use train code file)
3. Inference - Get the predictions on the test images of colon data using the trained models (use the inference-colon code file)
4. Performance Evaluation - Calculate the performance metrics for the predicted data (use the performancemetrics-colon code file)
