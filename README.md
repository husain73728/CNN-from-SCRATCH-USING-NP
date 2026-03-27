# 🧠 Convolutional Neural Network (CNN) from Scratch

## 📌 Overview
This project implements a **Convolutional Neural Network (CNN)** entirely from scratch using **NumPy**, without relying on deep learning frameworks like TensorFlow or PyTorch.

The objective is to build a deep understanding of:
- Convolution operations
- Max pooling
- Forward and backward propagation in CNNs
- Manual gradient computation and parameter updates

The model is trained on the **MNIST dataset** for handwritten digit classification.

---

## 🚀 Features
- Convolutional layers implemented from scratch
- Manual kernel (filter) operations
- Max pooling layer
- ReLU activation
- Fully connected (dense) layers
- Softmax output for multi-class classification
- Cross-entropy loss
- Backpropagation through convolution and pooling layers

---

## 🧱 Model Architecture
| Layer                  | Description                     |
|------------------------|---------------------------------|
| Input                  | 28×28 grayscale image           |
| Convolution Layer      | Multiple 3×3 filters            |
| ReLU Activation        | Non-linearity                   |
| Max Pooling            | Downsampling                    |
| Convolution Layer      | Feature extraction              |
| ReLU Activation        | Non-linearity                   |
| Max Pooling            | Downsampling                    |
| Flatten                | Convert to vector               |
| Dense Layer            | Fully connected                 |
| Output Layer           | Softmax (10 classes)            |

---

## 📊 Dataset
The model uses the **MNIST dataset**, containing:
- 60,000 training images
- 10,000 testing images
- 10 classes (digits 0–9)

Each image is:
- Grayscale
- Size: 28×28 pixels

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/cnn-from-scratch.git
cd cnn-from-scratch
pip install numpy scikit-learn
