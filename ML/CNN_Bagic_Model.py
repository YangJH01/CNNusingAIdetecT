import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/blues/Downloads/static_modified.csv')

X = df.drop(columns=['filename', 'sha256', 'family']).values.astype(np.float32)
y = df['family'].values.astype(np.float32)

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_train_tensor = torch.tensor(X_train).unsqueeze(1)  # (batch_size, 1, features)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)
X_val_tensor = torch.tensor(X_val).unsqueeze(1)
y_val_tensor = torch.tensor(y_val).unsqueeze(1)
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

class MalwareCNN(nn.Module):
    def __init__(self, input_size):
        super(MalwareCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MalwareCNN(input_size=X.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(epoch_loss)
    model.eval()
    y_val_pred = []
    y_val_true = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            y_val_pred.extend(preds)
            y_val_true.extend(y_batch.numpy())

    acc = accuracy_score(y_val_true, y_val_pred)
    print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {acc:.4f}")

losses = np.array(losses)

model.eval()
y_test_pred = []
y_test_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = (outputs.cpu().numpy() > 0.5).astype(int)
        y_test_pred.extend(preds)
        y_test_true.extend(y_batch.numpy())

test_acc = accuracy_score(y_test_true, y_test_pred)
print(f"\n최종 테스트 정확도: {test_acc:.4f}")

plt.plot(losses)
plt.show()

