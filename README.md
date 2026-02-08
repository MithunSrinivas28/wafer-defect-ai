
# Edge Vision AI
## Edge-Based Wafer Defect Detection Intelligence

Edge Vision AI is an end-to-end artificial intelligence system for automated wafer defect detection, explainability, and deployment on NXP edge devices. The project is designed for real-time, low-power, and scalable semiconductor quality inspection.


## 🔗 Project Resources
### [Dataset](https://drive.google.com/drive/folders/1PUNt98RTBxysWNoiKDndP9RSrii3a-W_?usp=drive_link)
### [Wafer Inspection Pipeline](https://drive.google.com/drive/folders/1tdnG_Y7DRLddYSVOkuULijsVjdHvX67F?usp=sharing)


## Problem Statement
Manual wafer inspection in semiconductor manufacturing is slow, costly, error‑prone, and difficult to scale for high‑volume production. Existing inspection systems lack real‑time intelligence and flexibility, creating a need for a lightweight, automated, and reliable solution.

## Proposed Solution
Edge Vision AI provides an intelligent inspection pipeline that:
- Automatically classifies wafer defects
- Runs efficiently on edge hardware
- Explains model predictions
- Learns from uncertain samples
- Operates in offline environments

---
# 🚀 Performance Summary (Key Highlights)

| Metric | Result |
|--------|--------|
| **Model Size** | **1.9 MB** |
| **Accuracy** | **75%+ overall** |
| **Training Accuracy** | ~69% |
| **Validation Accuracy** | ~79% |
| **Inference Speed** | Real-time |
| **Memory Usage** | Low |
| **Deployment** | Edge-ready (NXP compatible) |


---

## Key Innovations

- **Lightweight Edge AI**
  - Compact, embedded‑ready CNN  
  - Low‑latency, real‑time inference  

- **Self‑Learning System**
  - Automatically stores low‑confidence cases  
  - Model keeps improving with new data  

- **Edge Deployment**
  - INT8 quantized model  
  - Runs on NXP hardware with eIQ support  

- **Smart Analytics**
  - Batch inspection + yield calculation  
  - CSV logging with offline support  


## Model Accuracy and Validation Performance

- **Training Accuracy**: ~69%  
- **Validation Accuracy**: ~79%  

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

---

## Technology Stack
- **Machine Learning**: TensorFlow, Keras, NumPy
- **Image Processing**: OpenCV
- **Data Management**: Pandas, CSV
- **Deployment**: TensorFlow Lite, NXP eIQ Toolkit, C/C++
- **Development Tools**: Python, Google Colab, GitHub

---

## Project Structure


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




