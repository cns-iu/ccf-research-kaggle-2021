# Robust and Generalizable Segmentation of Human Functional Tissue Units

The Human BioMolecular Atlas Program aims to compile a reference atlas for the healthy human adult body at the cellular level. Functional tissue units (FTU, e.g., renal glomeruli and colonic crypts) are of pathobiological significance and relevant for modeling and understanding disease progression. Yet, annotation of FTUs is time consuming and expensive when done manually and existing algorithms achieve low accuracy and do not generalize well. This paper compares the five winning algorithms from the “[Hacking the Kidney](https://www.kaggle.com/c/hubmap-kidney-segmentation)” Kaggle competition to which more than a thousand teams from sixty countries contributed. We compare the accuracy and performance of the algorithms on a large-scale renal glomerulus Periodic acid-Schiff stain dataset and their generalizability to a colonic crypts hematoxylin and eosin stain dataset. Results help to characterize how the number of FTUs per unit area differs in relationship to their position in kidney and colon with respect to age, sex, BMI, and other clinical data and are relevant for advancing pathology, anatomy, and surgery.

The repo is structured in the following way:
```
├── images
├── models
│   ├── 1-Tom
│   └── 2-Gleb
│   └── 3-Whats goin on
│   └── 4-Deeplive.exe
│   └── 5-Deepflash2
├── supporting-information
├── utils
```
## Data

The colon data as well as the trained models can be accessed [here](https://drive.google.com/drive/u/0/folders/1m_NjuladAGt0iboq_eXHBRwjmkP2w6uc).

The kidney data is provided as a [HuBMAP collection](https://portal.hubmapconsortium.org/browse/collection/4964d24bbc6668a72c4cbb5e0393a6bc
).

## Models

The repository contains 5 models:
1. Tom (1st prize)
2. Gleb (2nd prize)
3. Whats goin on (3rd prize)
4. Deeplive.exe (1st Judges prize)
5. Deepflash2 (2nd Judges prize)


