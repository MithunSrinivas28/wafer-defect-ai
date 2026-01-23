# Chippenheimer Labs — Wafer Defect Detection using Edge AI

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

## Project Structure

