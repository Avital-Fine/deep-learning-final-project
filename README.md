# Pneumonia Detection: CNN vs Vision Transformer (ViT)

**Authors:** Avital Fine (ID: 208253823), Noa Lazar (ID: 322520339)  
**Course:** Deep Learning, 2025-Semester B, RUNI  

---

## Project Overview

This project compares two deep learning architectures for binary classification of pneumonia from chest X-ray images:

- **CNN (Convolutional Neural Network):** Standard VGG-like model for local feature extraction.  
- **ViT (Vision Transformer):** Transformer-based architecture capturing global context.  

We analyze model performance, training behavior, and generalization on a relatively small dataset.

---

## Dataset

- **Source:** [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Dataset sizes:**  
  - Training set: ~5,200 images  
  - Validation set: 16 images  
  - Test set: ~600 images  
- **Preprocessing:** Grayscale X-rays resized to 224×224 pixels and normalized to [0,1].  

**Note:** The validation set is very small compared to the test set, which made early stopping and hyperparameter tuning challenging.

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
| ViT   | 0.78     | 0.78     | 0.77   | 0.75     |

**Notes:**  
- **CNN:** ~404k parameters, faster convergence, strong baseline for small datasets.  
- **ViT:** ~14.4M parameters, longer training, captures global context, but underperforms CNN on this dataset.

---

## Challenges

Key challenges in this project included:  
1. Preventing overfitting given the relatively small dataset.  
2. Ensuring fair comparison by keeping training procedures consistent across models.  
3. Dealing with computational efficiency and resource constraints, as ViT required significantly more time and memory to train.  
4. Working with a very small validation set (16 images) compared to the test set (600 images), which made early stopping and hyperparameter tuning more difficult.

---

## Usage

Clone the repository:  

```bash
git clone https://github.com/Avital-Fine/deep-learning-final-project.git
cd deep-learning-final-project
