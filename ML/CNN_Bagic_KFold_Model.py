import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 데이터 불러오기
df = pd.read_csv('C:/Users/blues/Downloads/static_modified.csv')

# 입력 특징(X)과 라벨(y) 나누기
X = df.drop(columns=['filename', 'sha256', 'family']).values.astype(np.float32)
y = df['family'].values.astype(np.float32)

# 데이터 정규화
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# KFold 교차검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# CNN 모델 정의
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

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 손실 함수와 옵티마이저
criterion = nn.BCELoss()
optimizer = optim.Adam

# 교차검증 루프
all_test_acc = []
best_model = None  # 교차검증 후 가장 좋은 모델을 저장할 변수

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/{kf.get_n_splits()}")

    # 데이터 분할
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Tensor로 변환
    X_train_tensor = torch.tensor(X_train).unsqueeze(1)  # (batch_size, 1, features)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val).unsqueeze(1)

    # DataLoader
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    # 모델 초기화
    model = MalwareCNN(input_size=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    epochs = 30
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
        # 검증 정확도 측정
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

    # 최종 테스트 정확도 측정
    model.eval()
    y_test_pred = []
    y_test_true = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            y_test_pred.extend(preds)
            y_test_true.extend(y_batch.numpy())

    test_acc = accuracy_score(y_test_true, y_test_pred)
    print(f"\nFold {fold+1} - Final Test Accuracy: {test_acc:.4f}")
    all_test_acc.append(test_acc)

    if best_model is None or test_acc > max(all_test_acc):
        best_model = model.state_dict()

# 교차검증 결과 출력
print(f"\n교차검증 평균 정확도: {np.mean(all_test_acc):.4f}")
print(f"교차검증 정확도 표준편차: {np.std(all_test_acc):.4f}")

torch.save(best_model, 'malware_cnn_model.pth')