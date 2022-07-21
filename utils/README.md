### Utilities

- `Computation of FTU Density for HuBMAP Data and Visualizations.ipynb`: Used to compute FTU density for HuBMAP data and generate related visualizations. Requires all FTU masks and Anatomical Structure masks (kidney) in json format.
- `Performance Metrics - Colon`: Used to calculate all performance metrics on colon data. Requires a `submission.csv` file generated after running inference on a trained model.
- `Performance Metrics - Kidney`: Used to calculate all performance metrics on kidney data. Requires a `submission.csv` file generated after running inference on a trained model.
- `Result Analysis - Gloms Missed A3.ipynb`: Sample code to count FTUs and calculate missed and wrongly predicted FTUs (using area overlap, not used for final results).
- `Result Analysis - Gloms missed.ipynb`: Sample code to count FTUs and calculate missed and wrongly predicted FTUs (naive approach, not used for final results).
- `Result Analysis - Plot and Save Masks.ipynb`: Code to plot annotation masks and save them.
- `Result Analysis - glom_counter.py`: Script used to count the number of FTUs. Requires a `json` file of masks.
- `Result Analysis - mask_to_polygon.py`: Script used to convert masks from `*.png` to `*.json`.
- `Visualization - annotation_compare_viz.py`: Script used for predicted segmentation analysis (counting the false positives, false negatives, true positives) and generate a visualization to overlay and compare two masks.
- `Visualization - kaggle_violin_viz.py`: Script used to generate the violin plots for results.
