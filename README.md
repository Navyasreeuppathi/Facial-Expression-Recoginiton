# Facial Emotion Recognition using VGG16

A deep learning-based **Facial Emotion Recognition (FER)** system that classifies human facial expressions into seven emotion categories using **VGG16 Transfer Learning** and the **CK+48 dataset**.

**Validation Accuracy: 90.16%**

---

## Overview

Facial Emotion Recognition (FER) is a computer vision task that identifies human emotions from facial expressions.

This project uses **VGG16**, a Convolutional Neural Network (CNN) pretrained on **ImageNet**, as the base model. Transfer learning and fine-tuning are used to adapt VGG16 for facial emotion classification.

The model classifies facial expressions into seven categories:

* Anger
* Contempt
* Disgust
* Fear
* Happy
* Sadness
* Surprise

The trained model achieved a **90.16% validation accuracy** on the CK+48 dataset.

---

## Objectives

* Develop a deep learning model for facial emotion classification.
* Apply transfer learning using the VGG16 architecture.
* Classify facial expressions into seven emotion categories.
* Use data augmentation to improve model generalization.
* Fine-tune selected pretrained VGG16 layers.
* Evaluate the model using validation data.
* Predict emotions from custom and unseen facial images.
* Generate confidence scores for predictions.

---

## Emotion Classes

| No. | Emotion  |
| --- | -------- |
| 1   | Anger    |
| 2   | Contempt |
| 3   | Disgust  |
| 4   | Fear     |
| 5   | Happy    |
| 6   | Sadness  |
| 7   | Surprise |

---

## Dataset

### CK+48 Dataset

The project uses the **CK+48 (Extended Cohn-Kanade)** facial expression dataset for training and validation.

The dataset is organized into seven emotion categories:

```text
CK+48/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── sadness/
└── surprise/
```

An **80:20 training-validation split** was used.

> **Note:** The dataset is not included in this repository.

---

## Model Architecture

This project uses **VGG16 pretrained on ImageNet** as the base model.

The pretrained VGG16 feature extraction layers are combined with custom fully connected layers for emotion classification.

### Architecture

```text
Input Image
     ↓
224 × 224 × 3
     ↓
VGG16 (ImageNet Pretrained)
     ↓
Flatten
     ↓
Dense (512 neurons, ReLU)
     ↓
Dropout (0.5)
     ↓
Dense (256 neurons, ReLU)
     ↓
Dropout (0.5)
     ↓
Dense (7 neurons, Softmax)
     ↓
Predicted Emotion
```

---

## Transfer Learning and Fine-Tuning

Training was performed in two stages.

### Stage 1 — Feature Extraction

Initially, all pretrained VGG16 layers were frozen and only the newly added classification layers were trained.

| Configuration                | Value     |
| ---------------------------- | --------- |
| VGG16 Layers                 | Frozen    |
| Custom Classification Layers | Trainable |
| Learning Rate                | `1e-4`    |

### Stage 2 — Fine-Tuning

After the initial training stage, the **last four layers of VGG16** were unfrozen and fine-tuned using a smaller learning rate.

| Configuration                | Value     |
| ---------------------------- | --------- |
| Most VGG16 Layers            | Frozen    |
| Last Four VGG16 Layers       | Trainable |
| Custom Classification Layers | Trainable |
| Learning Rate                | `1e-5`    |

Fine-tuning allows the pretrained network to adapt its learned visual features to the facial expression recognition task.

---

## Training Configuration

| Parameter                 | Value                    |
| ------------------------- | ------------------------ |
| Base Model                | VGG16                    |
| Pretrained Weights        | ImageNet                 |
| Input Image Size          | 224 × 224                |
| Input Channels            | 3 (RGB)                  |
| Batch Size                | 8                        |
| Number of Classes         | 7                        |
| Training-Validation Split | 80:20                    |
| Optimizer                 | Adam                     |
| Loss Function             | Categorical Crossentropy |
| Hidden Layer Activation   | ReLU                     |
| Output Activation         | Softmax                  |
| Dropout                   | 0.5                      |
| Data Augmentation         | Yes                      |

---

## Data Augmentation

Data augmentation was applied to the training images to improve model generalization and reduce overfitting.

The following transformations were used:

* Rotation
* Width shifting
* Height shifting
* Zoom
* Horizontal flipping

Pixel values were normalized using:

```python
rescale = 1./255
```

---

## Training Callbacks

The following callbacks were used during training:

### Early Stopping

Early stopping was used to stop training when validation performance stopped improving and restore the best-performing model weights.

### Reduce Learning Rate

The learning rate was reduced when validation loss stopped improving, allowing the model to make smaller and more effective updates.

### Model Checkpoint

The best-performing model based on validation accuracy was automatically saved during training.

---

## Results

The trained VGG16 model achieved:

### Validation Accuracy

**90.16%**

Training and validation accuracy/loss curves were generated to monitor the learning process and evaluate model performance.

### Accuracy Plot

![Training and Validation Accuracy](images/accuracy_plot.png)

### Loss Plot

![Training and Validation Loss](images/loss_plot.png)

---

## Emotion Prediction

The trained model can be used to predict emotions from new facial images.

### Prediction Pipeline

```text
Input Facial Image
        ↓
Resize to 224 × 224
        ↓
Normalize Pixel Values
        ↓
VGG16 Model
        ↓
Softmax Classification
        ↓
Predicted Emotion
        ↓
Confidence Score
```

For each input image, the model provides:

* Predicted emotion
* Confidence score
* Probability for each emotion category

### Example

```text
Predicted Emotion : Surprise
Confidence        : 50.37%
```

### Prediction Example

![Emotion Prediction Example](images/prediction_example.png)

---

## Saved Model

The trained model is saved in Keras format:

```text
ckplus_final_vgg16.keras
```

The model can be loaded using:

```python
from tensorflow.keras.models import load_model

model = load_model("ckplus_final_vgg16.keras")
```

> The trained model file may be excluded from the GitHub repository depending on file-size limitations.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* VGG16
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn
* Pillow
* Google Colab

---

## Project Structure

```text
Facial-Emotion-Recognition/
│
├── README.md
├── Facial_Emotion_Recognition_VGG16.ipynb
├── requirements.txt
├── .gitignore
│
└── images/
    ├── accuracy_plot.png
    ├── loss_plot.png
    └── prediction_example.png
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Navyasreeuppathi/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```

### 2. Open the Notebook

Open:

```text
Facial_Emotion_Recognition_VGG16.ipynb
```

The notebook can be run using **Google Colab** or **Jupyter Notebook**.

### 3. Prepare the Dataset

Place the CK+48 dataset in the required directory:

```text
/content/CK+/CK+48/
```

Make sure the seven emotion folders are present:

```text
CK+48/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── sadness/
└── surprise/
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Model

Run the training cells in the notebook.

The model is trained using a two-stage approach:

1. VGG16 feature extraction
2. Custom classification layers
3. Data augmentation
4. VGG16 fine-tuning
5. Training callbacks

### 6. Test the Model

Upload a new facial image and run the prediction section of the notebook.

The model will display:

* Predicted emotion
* Confidence score
* Probability for each emotion category

---

## Requirements

The main libraries required are:

```text
tensorflow
keras
numpy
matplotlib
opencv-python
scikit-learn
Pillow
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

* Testing with larger and more diverse facial expression datasets.
* Comparing VGG16 with architectures such as ResNet and EfficientNet.
* Improving model generalization to real-world facial images.
* Deploying the model as a web or mobile application.

---

## Project Highlights

* Facial Emotion Recognition using Deep Learning
* VGG16 Transfer Learning
* CK+48 Dataset
* 7 Emotion Classes
* Two-Stage Training and Fine-Tuning
* Data Augmentation
* Early Stopping
* Learning Rate Reduction
* Model Checkpointing
* **90.16% Validation Accuracy**
* Custom Image Prediction
* Confidence Score for Predictions

---

## Author

**U. Navya Sree**

Computer Science Engineering Student

---

## Acknowledgements

* **CK+ Dataset** for providing facial expression images.
* **VGG16 and ImageNet** pretrained weights for transfer learning.
* **TensorFlow and Keras** for deep learning implementation.
* **Google Colab** for the training environment.

---

## License

This project is intended for educational and research purposes.
