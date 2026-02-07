# HASYv2 Hand Digit Recognition

## Overview
This repository contains the notebook implementation for classifying handwritten symbols from the HASYv2 dataset (369 classes) using PyTorch. The work focuses on data preparation, augmentation, and multiple CNN architectures, with a final high-accuracy CNN selected based on validation performance.

## Contributors
- Shaik Abdus Sattar (Seventie)
- A C Sanhitha Reddy ([@sanhithaac](https://github.com/sanhithaac))

## Dataset
HASYv2 Handwritten Symbol Dataset:
https://data.niaid.nih.gov/resources?id=zenodo_259444

The notebook loads:
- metaData.csv for dataset metadata
- train.csv for labeled training samples
- test.csv for unlabeled test samples

## Notebook
Main notebook: hasyv2-handwritten-symbols-dataset-pytorch (3).ipynb

Kaggle notebook: https://www.kaggle.com/code/shaikabdussattar/hasyv2-handwritten-symbols-dataset-pytorch

## Methodology
### Data understanding and preparation
- Inspect CSV files for columns, missing values, and label distribution.
- Map non-contiguous label IDs to a contiguous 0-368 range (369 classes) for training, then reverse-map predictions for submission.
- Convert images to grayscale tensors for model input.

### Data augmentation
To improve class balance and generalization, augmentation is applied primarily to classes with fewer than 100 samples, expanding them to roughly 200 samples per class. Techniques include:
- Random rotation
- Random affine transforms (shear and scale)
- Horizontal flip

### CNN architectures evaluated
1. Baseline CNN
   - Conv layers: 1→32, 32→64, 64→128
   - Max pooling after each conv
   - Fully connected head: 4×4×128 → 128 → 369

2. BatchNorm + Adaptive Pooling CNN
   - Conv layers: 1→32, 32→64, 64→128 with BatchNorm
   - Max pooling and AdaptiveAvgPool to 4×4
   - Fully connected head: 4×4×128 → 256 → 128 → 369
   - Dropout between FC layers

3. Final high-accuracy CNN (selected model)
   - Conv layers: 1→64, 64→128, 128→256 with BatchNorm
   - Max pooling and AdaptiveAvgPool to 4×4
   - Fully connected head: 4×4×256 → 512 → 256 → 128 → 369
   - Dropout (0.5 and 0.3) to reduce overfitting

### Training and optimization
- Loss: CrossEntropyLoss
- Final optimizer: SGD with momentum and weight decay
- Learning rate scheduling: StepLR
- Training loop tracks loss and accuracy per epoch with validation monitoring
- Methodology.pdf documents additional optimizer and learning-rate experiments (SGD, Adam, RMSprop) used to improve convergence

### Evaluation and inference
- Track training/validation accuracy and loss to monitor convergence.
- Generate predictions on test data, then reverse-map labels back to original IDs for submission.
- Save output as submission.csv.

## Additional resources
- Methodology.pdf for the full methodology write-up
