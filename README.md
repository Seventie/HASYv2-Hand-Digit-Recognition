# ✋ HASv2 Hand Digit Recognition
### 🏆 1st Place Winning Model — College Competition

## 🚀 Overview
This project contains my first-prize winning solution for the HASv2 Handwritten Digit Recognition competition.  
I built a custom Convolutional Neural Network (CNN) to classify digits (0–9) with preprocessing, augmentation, training visualization, and a final saved model.

---

## ⭐ Features
- Custom CNN architecture  
- Data preprocessing & augmentation  
- Training & validation accuracy/loss curves  
- Saved model for inference  
- Clean Jupyter notebook with explanations  
- High accuracy — winner of the competition

---

## 📁 Repository Structure
├── notebooks/
│ └── HASv2_Digit_Recognition.ipynb
│
├── models/
│ └── best_model.h5
│
├── results/
│ ├── accuracy_curve.png
│ ├── loss_curve.png
│ └── predictions.png
│
├── README.md
└── requirements.txt

📊 Dataset

Uses the HASv2 Hand Digit Dataset (grayscale digit images 0–9).

Preprocessing includes:

Normalization

Resizing

Augmentation (rotation, zoom, shifts, shear)

🧠 Model Architecture

A simple but effective CNN:

Conv2D → ReLU

MaxPooling

Dropout

Flatten → Dense → Softmax

📈 Training Performance

See the results/ folder for:

accuracy_curve.png

loss_curve.png

predictions.png

▶️ How to Run

Install dependencies:

pip install -r requirements.txt


Launch notebook:

jupyter notebook


Open:

notebooks/HASv2_Digit_Recognition.ipynb

🤖 Example Inference
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("models/best_model.h5")

img = cv2.imread("sample.png", 0)
img = cv2.resize(img, (28, 28)) / 255.0
img = img.reshape(1, 28, 28, 1)

print("Predicted Digit:", model.predict(img).argmax())
