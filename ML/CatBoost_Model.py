from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 로딩
df = pd.read_csv("..\..\keeping\static_modified.csv")

# 전처리
df = df.drop(columns=["filename", "sha256"])
df["label"] = df["family"].apply(lambda x: 0 if x == 0 else 1)
X = df.drop(columns=["label", "family"])
y = df["label"]

# 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 모델 정의 및 학습
model = CatBoostClassifier(verbose=100, iterations=100, depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 모델 저장 (선택)
# model.save_model("./models/catboost_static_model.cbm")
