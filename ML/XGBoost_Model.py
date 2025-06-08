import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. 데이터 로드
df = pd.read_csv('..\..\keeping\static_modified.csv')

# 2. 전처리
# 식별자 제거
df = df.drop(columns=['filename', 'sha256'])

# family → label (0: 정상, 1: 악성)
df['label'] = df['family'].apply(lambda x: 0 if x == 0 else 1)
df = df.drop(columns=['family'])

#XGBoost는 수치보다는 이 값보다 크냐 작냐를 보는 방식이라 따로 이상치 제거를 하지 않고 진행

# 3. 독립변수 / 종속변수 분리
X = df.drop(columns=['label'])
y = df['label']

# 4. train/test 분할
# train 80 test 20 / 재현 가능한 결과 위해서 랜덤시드 고정함
# label 분포를 train/test 동일하게 유지시키기 위해서 stratify = y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 모델 정의
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    tree_method='hist'
)

# 6. 학습
model.fit(X_train, y_train)

# 7. 평가
y_pred = model.predict(X_test)
print("📋 Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. 모델 저장 (선택)
# joblib.dump(model, './models/xgboost_static_model.joblib')
