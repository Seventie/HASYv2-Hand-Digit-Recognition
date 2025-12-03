# ✋ HASv2 Hand Digit Recognition
### 🏆 1st Place Winning Model — College Competition

## 🚀 Overview
This project contains my first-prize winning solution for the HASv2 Handwritten Digit Recognition competition.  
I experimented with multiple deep-learning architectures and finally built a **high-performance CNN with BatchNorm, Adaptive Pooling, and multi-layer fully connected blocks**.  
This final architecture provided the **best accuracy** and became the winning model.

---

## ⭐ Features
- Custom high-capacity CNN  
- Batch Normalization + Dropout for stability  
- Adaptive Avg Pooling for size-invariance  
- Multi-layer fully connected head  
- Training & validation accuracy/loss curves  
- Clean well-documented notebook  
- Final saved model for inference  
- **Winner of the college competition**

---

## 📊 Dataset
Uses the HASv2 Hand Digit Dataset (grayscale digit images 0–9).

Preprocessing includes:
- Normalization  
- Resizing  
- Augmentation (rotation, zoom, shifts, shear)

---

## 🧠 Model Architecture (Final Winning Model)

I tried multiple architectures, but the **best performing one** — the one that won the competition — is the following:

### ✅ **Final ConvNet (Highest Accuracy Model)**  
- Conv2D(1 → 64) + BatchNorm + ReLU  
- MaxPool  
- Conv2D(64 → 128) + BatchNorm + ReLU  
- MaxPool  
- Conv2D(128 → 256) + BatchNorm + ReLU  
- MaxPool  
- **AdaptiveAvgPool2d → (4 × 4)**  
- Flatten  
- Fully Connected: 512 → 256 → 128 → 369  
- Dropout for regularization  
- Raw logits → CrossEntropyLoss

### 🏆 Why this architecture worked best
- **BatchNorm** stabilized deep training  
- **Adaptive Pooling** made the representation consistent  
- **Deeper FC layers** captured complex patterns  
- **Higher channel depth (64 → 128 → 256)** improved feature extraction  

This model gave the **highest leaderboard score** and became the final winning submission.



---

## 📎 Additional Resources

### 📘 Kaggle Notebook
You can check out the full Kaggle version of my notebook here:  
🔗 **Kaggle Notebook:** https://www.kaggle.com/code/shaikabdussattar/hasyv2-handwritten-symbols-dataset-pytorch

### 👤 Kaggle Profile
Follow me on Kaggle for more ML projects and competitions:  
🔗 **Kaggle Profile:** https://www.kaggle.com/seventie

---


