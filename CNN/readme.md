# 📘 MODEL GUIDE

AI 기반 악성코드 이진 분류 모델을 위한 전체 실행 가이드입니다. 본 가이드는 데이터 수집부터 모델 학습까지 전 과정을 상세히 설명합니다.

---

## 🧩 1. 데이터 수집 및 구성

### 📁 폴더 구조 예시

```
DATA/
 ├── benign/
 │     ├── file1
 │     ├── file2.vir
 │     ├── file3.bytes
 │     └── ...
 └── malware/
       ├── file4
       ├── file5.vir
       ├── file6.bytes
       └── ...
```

### 📌 설명

* 데이터는 별도로 수집하여야합니다.
* `.bytes`, `.vir`, `.exe` 등 확장자에 관계없이 PE 기반 또는 바이너리 기반 악성코드/정상 샘플을 `benign`, `malware` 폴더로 구분해서 저장합니다.
* 이후 모든 파이프라인은 이 구조를 기반으로 작동하므로 반드시 위와 같은 형식 준수 필요.

---

## 🔄 2. 이미지 변환 단계

### 🔧 실행 방식

```bash
python change_entropy.py
python change_svd_single.py
```

* 명령행 인자 없이 직접 실행합니다.
* 변환할 데이터 경로(`input_dir`) 및 저장할 경로(`output_dir`)는 **각각의 파일 내부에서 직접 수정**해야 합니다.

예시 (`change_entropy.py`, `change_svd_single.py` 내부):

```python
input_dir = "C:/workspace/SVD/DATA"
output_dir = "C:/workspace/SVD/ENTROPY"  # 또는 SVD
```

> ⚠️ 동일한 `DATA/` 구조를 입력으로 사용하며, 각각 ENTROPY, SVD 이미지로 변환하여 저장합니다.

---

## 🔀 3. 데이터 분할 (Train/Val/Test)

### 🔧 실행 방식

```bash
python SVD_SAME_SPLIT.py
```

* 명령행 인자 없이 실행합니다.
* 입력 경로(`svd_dir`, `ent_dir`) 및 출력 경로(`output_dir`)는 **`SVD_SAME_SPLIT.py` 내부에서 직접 수정**해야 합니다.

예시:

```python
svd_dir = "C:/workspace/SVD/SVD"
ent_dir = "C:/workspace/SVD/ENTROPY"
svd_out = "C:/workspace/SVD/RESULT_SVD"
ent_out = "C:/workspace/SVD/RESULT_ENTROPY"
```

### 📁 최종 데이터 분할 구조

```
RESULT/
 ├── train/
 │   ├── malware/
 │   └── benign/
 ├── val/
 │   ├── malware/
 │   └── benign/
 └── test/
     ├── malware/
     └── benign/
```

* `train`, `val`, `test`는 동일한 기준으로 분할되며, 각 폴더 내부에 `malware`, `benign` 폴더가 존재합니다.

---

## 🧠 4. 모델 학습 실행

메인 학습 스크립트인 `dual_branch_ablation_runner.py`를 다음과 같은 방식으로 실행합니다.

### 💻 실행 명령 예시

```bash
python dual_branch_ablation_runner.py \
  --exp_name single_adamw_se_backbone0 \
  --svd_root C:/workspace/SVD/RESULT_SVD \
  --entropy_root C:/workspace/SVD/RESULT_ENTROPY \
  --exp_dir C:/workspace/SVD/result \
  --attention \
  --attn_type se \
  --freeze_scheme 0:3 \
  --label_smoothing 0.05 \
  --dropout_fc 0.5 \
  --dropout_backbone 0.0 \
  --epochs 30 \
  --opt_type adamw \
  --early_stopping 9
```

> 📌 `--svd_root`와 `--entropy_root`는 분할된 디렉토리의 `최상위` 디렉토리를 기준으로 설정되며, 내부적으로 train, validation 및 test set도 자동으로 로딩됨

---

## ⚙️ 주요 옵션 설명

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--exp_name` | `str` | `'debug'` | 실험 이름. 결과 저장 폴더명을 지정할 때 사용. |
| `--attention` | `flag` | `False` | Attention 사용 여부 (`True` 시 사용). |
| `--attn_type` | `str` | `'default'` | Attention 종류 선택: `'default'`, `'cbam'`, `'se'`. |
| `--base_from` | `str` | `None` | 기존 실험의 best_model.pth 불러오기용 이름, 폴더로 지정함. |
| `--svd_dir` | `str` | 경로 | 분할된 SVD 이미지 데이터가 저장된 디렉터리 경로. |
| `--ent_dir` | `str` | 경로 | 분할된 Entropy 이미지 데이터가 저장된 디렉터리 경로. |
| `exp_dir` | `str` | `'C:\\workspace\\SVD\\result'` | 결과 저장 기본 디렉터리. |
| `--batch_size` | `int` | `32` | 배치 사이즈. |
| `--epochs` | `int` | `40` | 전체 학습 에폭 수. |
| `--lr` | `float` | `5e-5` | 학습률 (Learning Rate). |
| `--weight_decay` | `float` | `1e-4` | 가중치 감쇠 (정규화 항). |
| `--label_smoothing` | `float` | `0.02` | 라벨 스무딩 계수. |
| `--mixup` | `float` | `0.0` | MixUp 알파 값. `0`이면 미사용. |
| `--dropout_fc` | `float` | `0.5` | Fully Connected layer의 dropout 비율. |
| `--dropout_backbone` | `float` | `0.0` | Backbone 네트워크의 dropout 비율. |
| `--freeze_scheme` | `str` | `'0:0,10:1,20:2'` | 에폭마다 backbone unfreeze 레벨 조절 (단계적 fine-tuning), 0:3 으로 설정시 모든 레벨학습. |
| `--warmup` | `int` | `0` | 전체 backbone unfreeze 후 warm-up 에폭 수. |
| `--ema` | `flag` | `False` | EMA(지수이동평균) 적용 여부. |
| `--swa` | `int` | `0` | SWA 적용 여부. `0`이면 미사용, `N`이면 마지막 `N`에폭 동안 SWA. |
| `--batch_mode` | `flag` | `False` | 실험 리스트를 반복 실행하는 batch mode 여부. |
| `--early_stopping` | `int` (optional) | `7` (if used with no value) | 조기 종료 기능 활성화 및 patience 설정. |
| `--opt_type` | `str` | `'adamw'` | Optimizer 종류: `'adamw'`, `'adabelief'`, `'lookahead_adamw'`, `'lookahead_adabelief'`. |


---

## 📂 학습 결과 저장 구조

```
result/
 └── single_adamw_se_backbone0/  ← exp_name 기준 생성
      ├── best_model.pth         ← 검증 최고 성능 모델 저장
      └── test_report.txt        ← 테스트 결과 저장
```

결과 저장 예시
```
              precision    recall  f1-score   support

      benign     0.9603    0.9758    0.9680       868
     malware     0.9796    0.9665    0.9730      1044

    accuracy                         0.9707      1912
   macro avg     0.9700    0.9711    0.9705      1912
weighted avg     0.9709    0.9707    0.9707      1912


```


---

## ✅ 추가 팁

* `--svd_dir`, `--ent_dir`, `--exp_dir`은 원하는 경로로 설정해주세요.
* 인자 설정을 하지 않을경우 `Default` 값으로 지정됩니다.
* `--attention` 없이 실행하면 기본 CNN 구조(concat)로 학습합니다.
* `--early_stopping` 은 내부에서 생성 시 기준점을 `val_loss`와  `val_acc` 중 하나를 선택할 수 있습니다.
``` python
if cfg.get('early_stopping') is not None:
        early_stopper = EarlyStopping(
            patience=cfg['early_stopping'],
            verbose=True,
            delta=0.001,
            path=os.path.join(exp_dir, 'best_model.pth'),
            monitor='acc'  # accuracy 기준으로 모니터링
        )
```
* 실험 별로 `--exp_name`을 바꿔주면 결과가 각각의 디렉토리에 저장되어 실험 비교가 용이합니다.
---

