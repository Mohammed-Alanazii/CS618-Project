# Data-Efficient Learning from EMG Signals for Hand Gesture Recognition

A deep learning pipeline for classifying hand gestures from surface electromyography (sEMG) signals using a compact 1D ResCNN with adaptive mixup augmentation and majority-voting inference.

## Overview

This project tackles cross-subject hand gesture recognition from sEMG data. It trains a lightweight ResCNN (~647K parameters) on the [NinaPro DB2](https://ninapro.hevs.ch) dataset and uses segment-level majority voting to produce robust predictions.

| Gesture | Label | NinaPro ID |
|---------|-------|------------|
| Rest | 0 | 0 |
| Fist | 1 | 6 |
| Large Grasp | 2 | 17 |
| Wrist Pronation | 3 | 25 |
| Tripod | 4 | 38 |

## Results

| Metric | Value |
|--------|-------|
| Window-level accuracy | ~70% |
| Hard voting accuracy | ~90% |
| Soft voting accuracy | ~90% |

Majority voting over continuous gesture segments significantly outperforms window-level predictions by smoothing transient noise and motion artifacts.

## Pipeline

```
Raw sEMG (12-ch, 2 kHz)
  → Bandpass filter (20–450 Hz, 4th-order Butterworth)
  → Notch filter (50 Hz)
  → Channel-wise standardization
  → Sliding window (200 samples, 50-step → 75% overlap)
  → Class-balanced sampling
  → Gaussian noise + Adaptive mixup
  → ResCNN
  → Majority voting (hard / soft)
```

## Model Architecture

The ResCNN stacks three residual blocks with increasing channel depth and decreasing kernel size:

- **Stem:** `Conv1D(12 → 64, k=15)` → GELU → MaxPool
- **Block 1:** `ResBlock(64 → 64, k=15)` → MaxPool
- **Block 2:** `ResBlock(64 → 128, k=7)` → MaxPool
- **Block 3:** `ResBlock(128 → 256, k=3)` → MaxPool
- **Head:** AdaptiveAvgPool → Dropout(0.5) → Linear(256 → 5)

## Training

- **Optimizer:** AdamW (lr=2e-3, weight_decay=3e-3)
- **LR schedule:** ReduceLROnPlateau (factor=0.5, patience=10)
- **Loss:** Cross-entropy with label smoothing (0.1)
- **Early stopping:** Patience of 30 epochs
- **Augmentation:**
  - Gaussian noise (std=0.03)
  - Adaptive mixup: α=0.25 (epochs 1–25), α=0.05 (26–50), off thereafter

## Project Structure

```
├── config.yaml            # Hyperparameter configuration
├── requirements.txt       # Python dependencies
├── src/
│   ├── train.py           # Full training & evaluation pipeline
│   └── utils.py           # Device & seed utilities
├── data/
│   ├── subset/            # Preprocessed NinaPro .npy files (S{id}_emg/labels.npy)
│   └── data_cite.bib      # Dataset citation
├── notebooks/
│   └── EMG_Pipeline.ipynb # Interactive exploration notebook
├── results/               # Training curves, confusion matrices, visualizations
└── manuscript/                 # LaTeX source for the paper
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (uses data/subset by default)
python src/train.py

# Specify custom paths
python src/train.py --data_dir data/subset --output_dir results/best_model --seed 123
```

> [!NOTE]
> The preprocessed `data/subset/` directory contains `.npy` files for all 40 subjects. Each subject has an EMG array (`S{id}_emg.npy`) and a labels array (`S{id}_labels.npy`). If re-creating from the raw dataset, refer to the notebook for the preprocessing steps.

## Dataset

The [NinaPro DB2](https://ninapro.hevs.ch) dataset (Atzori et al., 2014) contains sEMG recordings from 40 intact subjects performing hand gestures. This project uses a 5-gesture subset (Rest, Fist, LargeGrasp, Wrist Pronation, Tripod) with 12 electrodes at 2 kHz.

```
@article{atzori2014electromyography,
  title={Electromyography data for non-invasive naturally-controlled
         robotic hand prostheses},
  author={Atzori, Manfredo and Gijsberts, Arjan and Castellini, Claudio
          and Caputo, Barbara and others},
  journal={Scientific Data},
  volume={1},
  pages={140053},
  year={2014}
}
```
Made with ❤️ by [Mohammed Alanazi] | [GitHub](https://github.com/Mohammed-Alanazii)
Special thanks to Prof. Yassine Bouteraa for his guidance and support throughout this project.
Special thanks to the NinaPro team for providing the dataset and to the open-source community for their invaluable tools and resources.
