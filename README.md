# Assignment4

## Sentiment Analysis using Feedforward Neural Networks

This project focuses on performing sentiment analysis on IMDb movie reviews using Feedforward Neural Networks (FNNs). You'll preprocess text data, train neural networks, apply regularization, and implement ensemble learning techniques to boost performance.

---

##  Objectives

- Case study of sentiment analysis (from textbook, pages 247–260).
- Text processing using `sklearn.feature_extraction.text`.
- Building and training FNNs with PyTorch.
- Implementing k-Fold Cross Validation manually.
- Applying Dropout Regularization.
- Using Ensemble Learning to improve model accuracy.

---

## Implementation Tasks

### 1. Data Preparation (2 points)
- Download and clean IMDb reviews.
- Transform text into TF-IDF vectors.
- Use 70% of data for training, 30% for testing.

### 2. Building an FNN (3 points)
- Define a feedforward neural network (FNN) classifier.
- Input: TF-IDF vector.
- Output: Binary sentiment prediction (1 = positive, 0 = negative).
- Tune architecture (layers, neurons) for best performance.

### 3. Training and Hyperparameter Tuning (3 points)
- Train baseline model.
- Tune learning rate and weight decay (L2 regularization).
- Aim for ≥ 90% accuracy.
- Compare with logistic regression from the textbook.
- Consider using `torch.optim.Adam` for better training performance.

### 4. Training with k-Fold Cross Validation (5 points)
- Implement k-Fold CV manually in PyTorch.
- Tune the number of folds (k).
- Compare accuracy and training time with/without k-Fold CV.

### 5. Dropout Regularization (7 points)

#### a. Single Dropout Model (2 points)
- Apply dropout to the baseline FNN.
- Tune dropout probabilities.
- Compare with non-regularized version.

#### b. Ensemble of Dropout Models (5 points)
- Train ≥ 5 different dropout models.
- Use bagging to create an ensemble.
- Compare performance with baseline FNN.

---

##  Requirements

- Python 3.x
- PyTorch
- scikit-learn
- NumPy
- Matplotlib (optional for plotting)

Install dependencies with:

```bash
pip install torch scikit-learn numpy matplotlib
