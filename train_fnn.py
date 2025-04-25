import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle
import time
from fnn import FNN

# === TOGGLE ===
use_kfold = True  # Change to False to run regular training

# Load data
with open('data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

X_train_tensor = torch.FloatTensor(X_train.toarray())
X_test_tensor = torch.FloatTensor(X_test.toarray())
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# Results Storage
fnn_result = None
log_result = None

# FNN Training
def train_standard():
    model = FNN(input_size=10000, hidden_sizes=[512, 256, 128])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    start = time.time()
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
    end = time.time()

    model.eval()
    with torch.no_grad():
        preds = (model(X_test_tensor) > 0.5).float()
        accuracy = (preds == y_test_tensor).float().mean().item()

    global fnn_result
    fnn_result = (accuracy * 100, end - start)
    print(f"FNN Accuracy: {fnn_result[0]:.2f}%, Time: {fnn_result[1]:.2f}s")

def train_with_kfold(k=5):
    X = X_train_tensor
    y = y_train_tensor

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{k}")

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        model = FNN(input_size=10000, hidden_sizes=[512, 256, 128])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        start = time.time()
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_fold)
            loss = criterion(output, y_train_fold)
            loss.backward()
            optimizer.step()
        end = time.time()

        model.eval()
        with torch.no_grad():
            preds = (model(X_val_fold) > 0.5).float()
            acc = (preds == y_val_fold).float().mean().item()
            print(f"Fold {fold + 1} Accuracy: {acc * 100:.2f}%")

        fold_accuracies.append(acc)
        fold_times.append(end - start)

    avg_acc = sum(fold_accuracies) / k
    avg_time = sum(fold_times) / k
    global fnn_result
    fnn_result = (avg_acc * 100, avg_time)
    print(f"\nAverage K-Fold Accuracy: {fnn_result[0]:.2f}%")
    print(f"Average Training Time per Fold: {fnn_result[1]:.2f}s")

# Run FNN training
if use_kfold:
    train_with_kfold(k=5)
else:
    train_standard()

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
start = time.time()
log_model.fit(X_train, y_train)
end = time.time()
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)
log_result = (log_acc * 100, end - start)

print(f"\nLogistic Regression Accuracy: {log_result[0]:.2f}%, Time: {log_result[1]:.2f}s")

# Final Summary
print("\nTask Summary")
if fnn_result:
    print(f"FNN:                 Accuracy = {fnn_result[0]:.2f}%, Time = {fnn_result[1]:.2f}s")
print(f"Logistic Regression: Accuracy = {log_result[0]:.2f}%, Time = {log_result[1]:.2f}s")
