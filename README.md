# Fashion MNIST CNN Classifier
### Microsoft AI – ML Researcher Assignment

A 6-layer Convolutional Neural Network (CNN) built with Keras that classifies
images from the Fashion MNIST dataset into 10 clothing categories.
Predictions are demonstrated on 4 sample test images.

---

## Project Structure

| File | Description |
|------|-------------|
| `cnn_fashion_mnist.py` | Python implementation (Keras + OOP class) |
| `cnn_fashion_mnist.R`  | R implementation (Keras for R) |
| `requirements.txt`     | Python dependencies |
| `README.md`            | This file |

---

## Model Architecture (6 Layers)

| # | Layer | Details |
|---|-------|---------|
| 1 | Conv2D + MaxPooling | 32 filters, 3×3, ReLU, 28×28 → 14×14 |
| 2 | Conv2D + MaxPooling | 64 filters, 3×3, ReLU, 14×14 → 7×7   |
| 3 | Conv2D             | 128 filters, 3×3, ReLU                 |
| 4 | Flatten            | Converts feature maps to 1D vector     |
| 5 | Dense + Dropout    | 256 units, ReLU, 50% dropout           |
| 6 | Dense (Output)     | 10 units, Softmax                      |

> *MaxPooling and Dropout are supporting operations within their parent layers,
> not counted as standalone layers. The 6 layers represent the learnable/structural stages.*

---

## Dataset

**Fashion MNIST** – 70,000 grayscale images (28×28 px) across 10 classes:

`T-shirt/top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot`

- 60,000 training images
- 10,000 test images
- Auto-downloaded by Keras on first run

---

## Setup & Installation

### Python

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fashion-mnist-cnn.git
cd fashion-mnist-cnn

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python cnn_fashion_mnist.py
```

### R

```r
# In RStudio:
install.packages("keras")
library(keras)
install_keras()     # installs TensorFlow backend
                    # For keras3: install.packages("keras3")

# Then run:
source("cnn_fashion_mnist.R")
```

---

## Expected Output

| File | Python | R |
|------|--------|---|
| Training curves | `training_history.png` | `training_history_r.png` |
| Predictions     | `sample_predictions.png` | `sample_predictions_r.png` |
| Saved model     | `fashion_cnn.h5` | `fashion_cnn_r.h5` |

- **Terminal** – model summary, per-epoch accuracy & loss, final test accuracy
- Prediction plots show true label, predicted label, and confidence score
- Green title = correct prediction · Red title = incorrect prediction

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | *(update after training)* |
| Epochs | 10 |
| Batch Size | 64 |
| Optimiser | Adam |
| Loss Function | Categorical Cross-Entropy |

---

## Reproducibility

Both implementations use a fixed random seed (`42`) for TensorFlow and NumPy
to ensure consistent results across runs.

---

## How Predictions Work

After training, the model runs `model.predict()` on 4 sample test images.
Each image receives a probability score across all 10 classes — the class
with the highest probability is taken as the final prediction.