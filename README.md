# Pneumonia Detection: CNN vs Vision Transformer (ViT)

**Authors:** Avital Fine (ID: 208253823), Noa Lazar (ID: 322520339)  
**Course:** Deep Learning, 2025-Semester B, RUNI  

---

## Project Overview

This project compares two deep learning architectures for binary classification of pneumonia from chest X-ray images:

- **CNN (Convolutional Neural Network)**: Standard VGG-like model for local feature extraction.  
- **ViT (Vision Transformer)**: Transformer-based architecture capturing global context.  

We aim to analyze model performance, training behavior, and generalization.

---

## Dataset

- **Source:** [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Preprocessing:** Grayscale X-rays resized to 224×224, normalized to [0,1].  
- **Channel Conversion:** Grayscale expanded to 3 channels for ViT compatibility.  

---

## Training Details

- **Optimizer:** Adam, learning rate = 1e-4  
- **Batch size:** 32  
- **Early stopping:** Monitored validation loss  
- **Epochs to converge:** CNN ~10, ViT ~15  
- **Platform:** Local macOS M4  

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|----------|--------|----------|
| CNN   | 0.85     | 0.86     | 0.85   | 0.85     |
| ViT   | 0.77     | 0.81     | 0.77   | 0.75     |

- **CNN:** ~404k parameters, faster convergence, strong baseline.  
- **ViT:** ~14.6M parameters, longer training, better recall, captures global context.

### Observations

- CNN is efficient and performs well for local features.  
- ViT generalizes better for diffuse patterns but requires more data and compute.  
- ViT attention maps are more interpretable for highlighting relevant lung regions.  

---

## Usage

Clone the repository:  
```bash
git clone https://github.com/Avital-Fine/deep-learning-final-project
