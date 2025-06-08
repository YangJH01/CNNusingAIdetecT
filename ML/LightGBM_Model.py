import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

ROOT = 3 # 0: 기본, 1: combine_basic 2: combine_preprocessed    


df = pd.read_csv("..\..\keeping\static_modified.csv")
df = df.drop(columns=["filename", "sha256"])
df["label"] = df["family"].apply(lambda x: 0 if x == 0 else 1)
X = df.drop(columns=["label", "family"])
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=10,
    num_leaves=128,
    min_data_in_leaf=50,
    force_col_wise=True,
    random_state=42,
    is_unbalance=True
)

#model.fit early_stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=1)  # <- 이게 verbose 역할을 합니다
    ]
)



# 예측 및 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
print(confusion_matrix(y_test, y_pred))
