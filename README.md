# Facial Emotion Recognition using VGG16

## Overview

Facial Emotion Recognition (FER) is a deep learning-based computer vision project that identifies human facial expressions from images.

This project uses Transfer Learning with VGG16, a pretrained Convolutional Neural Network (CNN), to classify facial expressions into seven emotion categories using the CK+48 dataset.

The trained model achieved a validation accuracy of **90.16%** and can be used to predict emotions from new facial images.

## Objectives

- Develop a deep learning model for facial emotion classification.
- Use Transfer Learning with the VGG16 architecture.
- Classify facial expressions into seven emotion categories.
- Apply data augmentation to improve model generalization.
- Fine-tune selected pretrained VGG16 layers.
- Evaluate the trained model using validation data.
- Predict emotions from custom and unseen facial images.

## Emotion Classes

The model classifies facial expressions into the following seven categories:

| No. | Emotion |
|-----|---------|
| 1 | Anger |
| 2 | Contempt |
| 3 | Disgust |
| 4 | Fear |
| 5 | Happy |
| 6 | Sadness |
| 7 | Surprise |

## Dataset

### CK+48 Dataset

The project uses the **CK+48 (Extended Cohn-Kanade)** dataset for training and validation.

The dataset contains facial expression images organized into seven emotion categories:

```text
CK+48/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── sadness/
└── surprise/
