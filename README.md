# Lightweight-Face-Based-Deepfake-Detection

Overview

Lightweight-Face-Based-Deepfake-Detection is a deep learning project designed to classify whether a given facial image is real or AI-generated (GAN-based).
With the rapid advancement of image synthesis technologies such as GANs and deepfakes, distinguishing between real and generated faces has become increasingly important for maintaining digital authenticity and trust.

This project focuses on:

Building lightweight and high-performance deepfake detection models

Comparing multiple architectures (CNN-based and Transformer-based)

Evaluating the effectiveness of localized facial region analysis (eyes, nose, mouth)

Assessing model robustness against retouched (Photoshopped) real images

Key Features

Binary Classification: Distinguish between real and GAN-generated faces

Multi-model Comparison: ResNet50, EfficientNet-B0, MobileNetV2, and Vision Transformer (ViT)

Region-based Experiments: Evaluate performance using cropped facial parts (eyes, nose, mouth)

Robustness Test: Analyze model behavior on retouched real images

Grad-CAM Visualization: Identify regions that models focus on during prediction

Lightweight Optimization: Balance between accuracy and computational efficiency

Project Structure
Lightweight-Face-Based-Deepfake-Detection/
├── dataset_통합/                # Real and Fake face datasets (Kaggle 140K)
├── EfficientNet/                # EfficientNet-B0 implementation
├── Mobilenet/                   # MobileNetV2 implementation
├── VIT/                         # Vision Transformer implementation
├── ResNet.ipynb                 # ResNet50 experiment notebook
├── report/                      
│   └── 6조_최종보고서.pdf        # Full project report (Korean)
└── README.md

Methodology
Dataset

Source: 140k Real and Fake Faces (Kaggle)

Composition:

Training: 50,000 real + 50,000 fake

Validation: 10,000 real + 10,000 fake

Test: 10,000 real + 10,000 fake

Preprocessing:

Normalization using dataset mean & std

Image augmentation:

RandomCrop(224), RandomHorizontalFlip()

ColorJitter(brightness, contrast, saturation, hue)

GaussianBlur(kernel_size=3)

Labeling via ImageFolder (fake: 0, real: 1)

Models and Training
Model	Backbone	Best Accuracy	Fine-tuning Strategy	Learning Rate	Weight Decay
ResNet50	CNN	95.0%	Full network fine-tuning	0.0001	0.0001
EfficientNet-B0	CNN	94.5%	Full fine-tuning	0.001	0.00001
MobileNetV2	CNN	92.9%	Full fine-tuning	0.0005	0.0005
Vision Transformer (ViT)	Transformer	96.3%	FC + BN	0.00001	0.00005

Optimizer: Adam

Loss Function: BCEWithLogitsLoss

Scheduler: CosineAnnealingLR

Batch Size: 64

Experimental Results
Main Experiment — Real vs GAN-generated Faces
Model	Accuracy	Recall	F1-score	Specificity	AUC
ResNet50	95.0%	95.2%	95.0%	94.8%	0.9882
EfficientNet-B0	94.5%	92.2%	94.3%	96.8%	-
MobileNetV2	92.9%	93.4%	92.9%	92.4%	0.9842
Vision Transformer	96.3%	96.8%	96.3%	95.8%	-

Additional Experiment 1 — Local Region (Eyes, Nose, Mouth)
Model	Region	Accuracy
ResNet50	Eyes + Nose	87.9%
EfficientNet-B0	Eyes + Nose	88.8%
MobileNetV2	Nose + Mouth	89.0%
Vision Transformer	Eyes	77.9%

Additional Experiment 2 — Retouched Real Images
Model	Accuracy	F1-score	Observation
ResNet50	91.0%	95.3%	Stable under moderate retouching
EfficientNet-B0	93.0%	96.3%	Most robust to color and tone changes
MobileNetV2	82.0%	90.1%	Slight drop under edited conditions
Vision Transformer	61.0%	75.8%	Sensitive to color correction & contour edits
