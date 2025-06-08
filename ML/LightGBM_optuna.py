import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import joblib

# 1. 데이터 로딩
df = pd.read_csv("..\..\keeping\static_modified.csv")
df = df.drop(columns=["filename", "sha256"])
df["label"] = df["family"].apply(lambda x: 0 if x == 0 else 1)
X = df.drop(columns=["label", "family"])
y = df["label"]

# 2. 튜닝용 train/valid 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 3. Optuna 튜닝
def objective(trial):
    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "is_unbalance": True,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    }

    model = lgb.LGBMClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    y_pred = model.predict(X_valid)
    return f1_score(y_valid, y_pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best F1 score:", study.best_value)
print("Best trial params:", study.best_trial.params)

# 4. 최종 학습용 Train/Test 분할
X_train_final, X_test, y_train_final, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. 최적 파라미터로 모델 학습
best_params = study.best_trial.params
best_model = lgb.LGBMClassifier(**best_params, is_unbalance=True)
best_model.fit(X_train_final, y_train_final)

# 6. 성능 평가
y_pred = best_model.predict(X_test)
print("\n[Test Set Performance]")
print(classification_report(y_test, y_pred, digits=4))
print(confusion_matrix(y_test, y_pred))

# 7. 모델 저장 (선택)
# joblib.dump(best_model, "./models/lightgbm_optuna_best.joblib")
