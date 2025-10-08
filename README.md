# Lightweight-Face-Based-Deepfake-Detection

## Project Overview

This project aims to detect whether a given facial image is **real** or **AI-generated (GAN-based)**.  
With the rapid development of deepfake and synthetic media, distinguishing authentic human faces from generated ones has become an important challenge.  
To address this, multiple deep learning models were implemented and compared with a focus on lightweight and high-performance architectures.

### Main Objectives
- Develop lightweight and efficient deepfake detection models  
- Compare CNN and Transformer-based architectures  
- Evaluate model performance on specific facial regions (eyes, nose, mouth)  
- Test robustness against image retouching and editing  

---

## Project Structure

```bash
Lightweight-Face-Based-Deepfake-Detection/
├── dataset_통합/               # Combined real/fake face dataset (Kaggle 140K)
├── EfficientNet/               # EfficientNet-B0 model and training scripts
├── Mobilenet/                  # MobileNetV2 implementation
├── VIT/                        # Vision Transformer implementation
├── ResNet.ipynb                # ResNet50 experiment notebook
├── report/
│   └── 6조_최종보고서.pdf       # Full project report (Korean)
└── README.md                   # Project documentation

---

## Dataset

- **Source:** [140k Real and Fake Faces (Kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Composition:**
  - Train: 50,000 real / 50,000 fake  
  - Validation: 10,000 real / 10,000 fake  
  - Test: 10,000 real / 10,000 fake
- **Preprocessing:**
  - Normalization using dataset mean and standard deviation  
  - Data augmentation techniques:
    - `RandomCrop(224)`
    - `RandomHorizontalFlip()`
    - `ColorJitter(brightness, contrast, saturation, hue)`
    - `GaussianBlur(kernel_size=3)`
  - Labeling handled via `ImageFolder` with mapping `{fake: 0, real: 1}`

---

## Methodology

### Model Architectures

| Model | Backbone | Best Accuracy | Fine-tuning Strategy | Learning Rate | Weight Decay |
|--------|-----------|---------------|----------------------|----------------|---------------|
| ResNet50 | CNN | 95.0% | Full fine-tuning | 0.0001 | 0.0001 |
| EfficientNet-B0 | CNN | 94.5% | Full fine-tuning | 0.001 | 0.00001 |
| MobileNetV2 | CNN | 92.9% | Full fine-tuning | 0.0005 | 0.0005 |
| Vision Transformer (ViT) | Transformer | **96.3%** | FC + BN | 0.00001 | 0.00005 |

**Training configuration**
- Optimizer: Adam  
- Loss function: BCEWithLogitsLoss  
- Scheduler: CosineAnnealingLR  
- Batch size: 64  

---

## Experimental Results

### 1. Real vs GAN-generated Faces

| Model | Accuracy | Recall | F1-score | Specificity | AUC |
|--------|-----------|---------|-----------|--------------|------|
| ResNet50 | 95.0% | 95.2% | 95.0% | 94.8% | 0.9882 |
| EfficientNet-B0 | 94.5% | 92.2% | 94.3% | 96.8% | - |
| MobileNetV2 | 92.9% | 93.4% | 92.9% | 92.4% | 0.9842 |
| Vision Transformer | **96.3%** | **96.8%** | **96.3%** | **95.8%** | - |

---

### 2. Region-based Analysis (Eyes, Nose, Mouth)

| Model | Region | Accuracy |
|--------|--------|-----------|
| ResNet50 | Eyes + Nose | 87.9% |
| EfficientNet-B0 | Eyes + Nose | 88.8% |
| MobileNetV2 | Nose + Mouth | **89.0%** |
| Vision Transformer | Eyes | 77.9% |

---

### 3. Robustness Test (Retouched Real Images)

| Model | Accuracy | F1-score | Observation |
|--------|-----------|-----------|-------------|
| ResNet50 | 91.0% | 95.3% | Stable under mild retouching |
| EfficientNet-B0 | **93.0%** | **96.3%** | Most robust to color/tone edits |
| MobileNetV2 | 82.0% | 90.1% | Slight degradation under edited images |
| Vision Transformer | 61.0% | 75.8% | Sensitive to global color and contour changes |

---
