# EDGE-VISION — Wafer Defect Detection using Edge AI

## Overview

This project implements a lightweight deep learning system for automated defect detection in semiconductor wafer inspection images. The model classifies microscope images into clean and defective categories, enabling faster quality control and reduced manual inspection in semiconductor manufacturing environments.

The solution is designed for edge deployment, prioritizing low latency, small model size, and high portability.

---

## Problem Statement

Modern semiconductor fabrication generates large volumes of high-resolution inspection images. Manual analysis and centralized cloud processing introduce latency, cost, and scalability limitations.

This project addresses these challenges by building an on-device AI model capable of detecting wafer defects in real time.

---

## Key Features

- Multi-class defect classification
- Lightweight CNN architecture for edge deployment
- Transfer learning using pretrained vision models
- Grayscale image preprocessing
- Exportable to ONNX/TFLite formats
- Modular training and evaluation pipeline

---

## Defect Categories

The dataset contains the following classes:

- Clean
- Crack
- Scratch
- Particle
- Void
- Bridge
- Other

(Exact categories may vary based on dataset availability.)

---

## Technology Stack

- Language: Python 3.9+
- Framework: PyTorch
- Vision Library: torchvision
- Image Processing: OpenCV, Pillow
- Model: MobileNetV2 / EfficientNet-B0
- Export: ONNX / TensorFlow Lite
- Version Control: Git, GitHub




---

## 🧠 Model Architecture

- Algorithm: CNN + MobileNetV3-Small (Transfer Learning)
- Base Model: Pretrained on ImageNet
- Custom Layers:
  - Global Average Pooling
  - Dense (128)
  - Dense (Number of Classes)

Hidden Layers: 1–2

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab

---

## ⚡ Training Configuration

- Activation (Hidden): ReLU
- Activation (Output): Softmax
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 10–20
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy

---

## 📏 Data Preprocessing

- Resize: 224 × 224
- Grayscale Conversion
- Normalization: /255 (0–1 range)
- Data Augmentation:
  - Rotation
  - Flip
  - Zoom

---

## 📈 Explainable AI

- Technique: Grad-CAM Heatmaps
- Purpose: Visualize defect regions
- Used during testing and demo
- Not included in deployed model

---

## 📦 Model Optimization

- Transfer Learning
- Quantization (INT8) for edge deployment
- Target Model Size: < 3 MB

---

## 🚀 Pipeline


---

## Project Structure

