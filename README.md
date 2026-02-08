
# Edge Vision AI
## Edge-Based Wafer Defect Detection Intelligence

WaferGuard AI is an end-to-end artificial intelligence system for automated wafer defect detection, explainability, and deployment on NXP edge devices. The project is designed for real-time, low-power, and scalable semiconductor quality inspection.

## Problem Statement
Manual wafer inspection in semiconductor manufacturing is:
- Time-consuming
- Error-prone
- Expensive
- Difficult to scale

Conventional inspection systems lack flexibility and real-time intelligence. Industries require a lightweight, automated, and reliable solution for high-volume production environments.

---

## Proposed Solution
WaferGuard AI provides an intelligent inspection pipeline that:
- Automatically classifies wafer defects
- Runs efficiently on edge hardware
- Explains model predictions
- Learns from uncertain samples
- Operates in offline environments

This enables consistent, accurate, and scalable quality control.

---

## System Overview
The system follows a modular pipeline:

```plaintext
Image Acquisition → Preprocessing → AI Inference → Confidence Evaluation
                                           ↓
                                Explainable AI Visualization
                                           ↓
                               Edge Deployment and Monitoring
```

Training and deployment pipelines are separated for optimization and reliability.

---

## Key Innovations
- **Lightweight AI Architecture**
  - Compact CNN model (~5 MB)
  - Optimized for embedded deployment
  - Low latency inference

- **Self-Learning Mechanism**
  - Automatic storage of low-confidence samples
  - Enables continuous improvement
  - Reduces manual labeling effort

- **Edge AI Deployment**
  - INT8 quantized model
  - Compatible with NXP eIQ toolkit
  - Real-time inference on microcontrollers

- **Smart Analytics**
  - Batch inspection system
  - Yield computation
  - CSV-based logging
  - Offline synchronization

---

## Technology Stack
- **Machine Learning**: TensorFlow, Keras, NumPy
- **Image Processing**: OpenCV
- **Explainability**: Grad-CAM, Matplotlib
- **Data Management**: Pandas, CSV
- **Deployment**: TensorFlow Lite, NXP eIQ Toolkit, C/C++
- **Development Tools**: Python, Google Colab, Jupyter, GitHub

---

## Project Structure
```plaintext
waferguard-ai
│
├── notebooks (Training and Pipeline)
├── models (Saved Models)
├── datasets (Self-learning data)
├── results (Logs and CSV files)
├── edge (NXP firmware files)
├── assets (Images and diagrams)
└── README.md
```

---

## System Workflow
### Step 1: Data Input
Wafer images are captured using camera modules or uploaded manually.

### Step 2: Preprocessing
Images are resized, normalized, and formatted for model input.

### Step 3: Inference
The CNN model predicts defect class and confidence score.

### Step 4: Confidence Evaluation
- High confidence → Accepted result
- Low confidence → Stored for retraining

### Step 5: Explainability
Grad-CAM generates visual explanations highlighting defect regions.

### Step 6: Edge Deployment
Optimized model runs on NXP hardware for real-time inspection.

---

## Performance Summary
- **Model Size**: 1.9 MB
- **Accuracy**: 75%+
- **Inference Speed**: Real-time
- **Memory Usage**: Low

Performance varies based on dataset and hardware configuration.

---
## Model Accuracy and Validation Performance

- **Training Accuracy**: ~69%  
- **Validation Accuracy**: ~70%  

### Definition

**Training Accuracy** represents how well the model learns patterns from the training dataset.  
It indicates the model’s ability to fit known data.

**Validation Accuracy** represents how well the trained model performs on unseen data.  
It measures the model’s generalization capability in real-world scenarios.

### Interpretation

The close alignment between training and validation accuracy indicates:

- Minimal overfitting  
- Stable learning behavior  
- Good generalization performance  

This balance confirms that the model performs consistently on both known and unknown wafer images.

## Impact

### Industrial Impact
- Reduced inspection costs
- Improved manufacturing yield
- Faster quality validation

### Technical Impact
- Promotes Edge AI adoption
- Improves model transparency
- Enables scalable deployment

### Innovation Impact
- Combines XAI and embedded AI
- Supports adaptive learning
- Bridges research and production

---

## Hackathon Highlights
- Complete end-to-end solution
- Edge-ready deployment
- Explainable decision system
- Self-learning pipeline
- Industry-focused design

This project demonstrates the transition from prototype to deployable system.

---

## Future Enhancements
- Automated retraining pipeline
- Cloud-based monitoring dashboard
- Live camera integration
- Multi-device deployment
- Predictive maintenance features

---

## Team
- [Khusi Mittal](https://github.com/khushimittal1209-afk)
- [Mithun Srinivas](https://github.com/MithunSrinivas28)


---

## Acknowledgements
- Hackathon Organizers
- NXP eIQ Platform
- Open Source Community

Thank you for reviewing WaferGuard AI.




