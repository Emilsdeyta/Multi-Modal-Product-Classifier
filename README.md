# Multi-Modal-Product-Classifier
A deep learning pipeline that combines product images and text metadata to classify fashion products into categories. Built with ResNet-18 for image encoding and DistilBERT for text encoding, fused into a single classification head.

## Dataset
- **Fashion Product Images (Small)** — Kaggle (paramaggarwal)
- 44,000+ products, 10 categories selected
- Each product includes: image (jpg) + metadata (gender, category, colour, season)

## Model Architecture

### 1. Image Encoder — ResNet-18
- Pretrained on ImageNet weights
- Final fully-connected layer removed
- Output: **512-dimensional feature vector**

### 2. Text Encoder — DistilBERT
- Pretrained `distilbert-base-uncased` model
- Only last 2 transformer layers fine-tuned, remaining layers frozen
- CLS token output used
- Output: **768-dimensional feature vector**

### 3. Fusion Strategy — Concatenation
- Image (512) + Text (768) → **1280-dimensional combined vector**
- Dense layers: 1280 → 256 → NUM_CLASSES
- Activation: ReLU, Dropout: 0.3

## Hyperparameters

| Parameter      | Value          |
|----------------|----------------|
| Image size     | 64×64          |
| Max token len  | 16             |
| Batch size     | 64             |
| Epochs         | 3              |
| Learning rate  | 3e-4           |
| Optimizer      | Adam           |
| Train samples  | 2,000          |

## Results

### Main Model (Multi-Modal)
| Metric       | Value  |
|--------------|--------|
| Macro-F1     | 0.8330 |
| Accuracy     | 1.00   |
| Val Loss     | 0.0101 |

### Modality Analysis

| Modality        | Macro-F1 |
|-----------------|----------|
| Text only       | 0.8330   |
| Image only      | 0.0692   |
| **Both**        | **0.8330** |

## Observations

Each modality was tested independently by zeroing out the other:

- The **text encoder** (DistilBERT) is responsible for nearly all of the model's performance on its own.
- The **image encoder** (ResNet-18) contributes negligible additional value — because the metadata text (`"Men Apparel Topwear Shirts Navy Blue"`) already unambiguously determines the category.
- In this setting, the CNN remains in the background. For images to play a meaningful differentiating role, the text input would need to be less informative.

## Training Dynamics

| Epoch | Train Loss | Train F1 | Val Loss | Val F1 |
|-------|------------|----------|----------|--------|
| 1     | 0.1531     | 0.6762   | 0.0059   | 0.8329 |
| 2     | 0.0040     | 0.8328   | 0.0051   | 0.8095 |
| 3     | 0.0073     | 0.8245   | 0.0101   | 0.8330 |

The model converged rapidly after the first epoch. The gap between train and val loss remains minimal — no overfitting observed.

## Technologies Used
- PyTorch, TorchVision
- HuggingFace Transformers (DistilBERT)
- scikit-learn (metrics)
- Pillow, pandas
