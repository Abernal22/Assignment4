import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import time
from fnn import FNN

#Load data
with open('data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

X_train_tensor = torch.FloatTensor(X_train.toarray())
X_test_tensor = torch.FloatTensor(X_test.toarray())
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

#train FNN
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

# Evaluate
model.eval()
with torch.no_grad():
    preds = (model(X_test_tensor) > 0.5).float()
    accuracy = (preds == y_test_tensor).float().mean().item()

print(f"FNN Accuracy: {accuracy*100:.2f}%, Time: {end - start:.2f}s")

# logistic regression
log_model = LogisticRegression(max_iter=1000)
start = time.time()
log_model.fit(X_train, y_train)
end = time.time()
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)

print(f"Logistic Regression Accuracy: {log_acc*100:.2f}%, Time: {end - start:.2f}s")
