"""
Input Folder 구조는
DATA
 ├── benign
 │      ├── file1.png
 │      ├── file2.png
 │      ├── file3.png
 │      └── ...
 └── malware
        ├── file4.png
        ├── file5.png
        ├── file6.png
        └── ...

와 같이 구성되어있다고 가정합니다.
"""

import os
import random
import shutil
from tqdm import tqdm

# 원본 루트 경로
SVD_ROOT = r"C:\workspace\SVD\new_data\new_every_color"
ENT_ROOT = r"C:\workspace\SVD\new_data\every_entropy"

# 출력 루트 경로
SVD_OUT = r"C:\workspace\SVD\new_data\new_every_color_split_same"
ENT_OUT = r"C:\workspace\SVD\new_data\new_every_entropy_split_same"

# 분할 비율
SPLIT_RATIO = (0.85, 0.10, 0.05)  # train, val, test

# 고정 시드 설정
random.seed(42)

# benign, malware 각각에 대해 처리
for label in ['benign', 'malware']:
    svd_path = os.path.join(SVD_ROOT, label)
    ent_path = os.path.join(ENT_ROOT, label)

    # 기준 리스트 생성 (확장자는 그대로 사용)
    files = sorted([f for f in os.listdir(svd_path) if f.lower().endswith(".png")])
    random.shuffle(files)

    total = len(files)
    train_end = int(total * SPLIT_RATIO[0])
    val_end = train_end + int(total * SPLIT_RATIO[1])

    split_mapping = {}
    for i, fname in enumerate(files):
        if i < train_end:
            split_mapping[fname] = 'train'
        elif i < val_end:
            split_mapping[fname] = 'val'
        else:
            split_mapping[fname] = 'test'

    # 디렉토리 생성
    for split in ['train', 'val', 'test']:
        for base in [SVD_OUT, ENT_OUT]:
            os.makedirs(os.path.join(base, split, label), exist_ok=True)

    # 파일 복사
    for fname in tqdm(files, desc=f"Copying {label}"):
        split = split_mapping[fname]

        # SVD 복사
        src_svd = os.path.join(svd_path, fname)
        dst_svd = os.path.join(SVD_OUT, split, label, fname)
        shutil.copy(src_svd, dst_svd)
        
        # Entropy 복사
        src_ent = os.path.join(ent_path, fname)
        dst_ent = os.path.join(ENT_OUT, split, label, fname)

        if os.path.exists(src_ent):
            shutil.copy(src_ent, dst_ent)
        else:
            print(f"⚠️ Entropy 이미지 없음: {src_ent}")
            
