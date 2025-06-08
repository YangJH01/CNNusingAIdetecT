import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from module import DualBranchModel, model_feature_extractor
from skimage.filters.rank import entropy
from skimage.morphology import disk

# 이거 기반으로 web에서 사용할 수 있도록 수정하면 될듯. baseline이라고 생각

# ======== 구성 ========
MODEL_PATH = r"../models/conv.pth" # 상대경로인데 절대경로로 바꿔도 상관없음
IMAGE_SIZE = (224, 224) # 여기는 바꾸면 안됨
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # cpu, gpu 선택
CLASS_NAMES = ['benign', 'malware']
MAX_FILE_MB = 100   # 최대 파일 크기 (MB)

# ======== 유틸 ========
def auto_width(length: int) -> int:
    cand = [4096, 3584, 3072, 2560, 2048, 1792, 1536, 1280,
            1024, 896, 768, 640, 512, 448, 384, 320,
            256, 224, 192, 160, 128, 96, 64, 48, 32]
    best, diff = None, float('inf')
    for w in cand:
        h = int(np.ceil(length / w))
        if h / w <= 2.0 and h <= 2048 and abs(h - w) < diff:
            best, diff = w, abs(h - w)
    return best or 4096

def read_pe_file(filepath: str) -> list[int]:
        with open(filepath, 'rb') as f:
            return list(f.read())

def is_file_too_large(path, max_mb):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return size_mb > max_mb, size_mb

def compute_adaptive_rank(height, width, alpha=0.1, k_min=20, k_max=100):
    base = int(min(height, width) * alpha)
    return max(k_min, min(k_max, base))

# ======== 변환 ========
def convert_to_svd_colormap(data: list[int]) -> np.ndarray:
    def pad_byte_array(byte_array, target_len):
        if not byte_array:
            return [0] * target_len
        last_val = byte_array[-1]
        return byte_array + [last_val] * (target_len - len(byte_array))

    length = len(data)
    width = auto_width(length)
    height = int(np.ceil(length / width))
    padded = pad_byte_array(data, width * height)
    matrix = np.array(padded, dtype=np.uint8).reshape((height, width)).astype(np.float32)

    # ✅ adaptive rank 계산
    k = compute_adaptive_rank(height, width)

    # ✅ SVD 압축 및 복원
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

    # ✅ 정규화 및 8bit 변환
    normed = (compressed - compressed.min()) / (compressed.max() - compressed.min() + 1e-5)
    gray = (normed * 255).astype(np.uint8)

    # ✅ Jet 컬러맵 적용
    jet = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # ✅ Resize
    jet = cv2.resize(jet, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    return jet


def convert_to_entropy_image(data: list[int]) -> np.ndarray:
    # 원본 shape 결정
    w = auto_width(len(data))
    h = int(np.ceil(len(data) / w))

    # 패딩 및 2D 행렬 변환
    padded = data + [0] * (w * h - len(data))
    gray = np.array(padded, dtype=np.uint8).reshape(h, w)

    # Entropy 계산 (resize 이전)
    ent = entropy(gray, disk(5)).astype(np.float32)

    # 정규화 (0~255 범위로)
    if ent.max() > 0:
        ent = (255 * (ent / ent.max())).astype(np.uint8)
    else:
        ent = np.zeros_like(ent, dtype=np.uint8)

    # Resize + RGB 변환
    ent_resized = cv2.resize(ent, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(ent_resized, cv2.COLOR_GRAY2RGB)


# ======== 전처리 ========
def preprocess(img: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# ======== 모델 추론 ========
def predict(svd_img, ent_img):
    model = DualBranchModel(model_feature_extractor, num_classes=2).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        out = model(svd_img.to(device), ent_img.to(device))
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    return prob

# ======== GUI 실행 ========
# 이부분은 쉽게 이해하기 위해 넣은거라 중요한건 위쪽부분
def run_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="파일선택", filetypes=[("PE 파일", "*.bytes *.bin *.exe *.dll"), ("All files", "*.*")])
    if not file_path:
        print("파일을 선택하지 않았습니다.")
        return

    too_big, size_mb = is_file_too_large(file_path, MAX_FILE_MB)
    if too_big:
        print(f"선택한 파일 크기: {size_mb:.2f}MB")
        print(f"{MAX_FILE_MB}MB를 초과하는 파일은 지원하지 않습니다.")
        return
    
    print(f"선택된 파일: {file_path}")
    print(f"파일 크기: {size_mb:.2f}MB")
    data = read_pe_file(file_path)
    svd_np = convert_to_svd_colormap(data)
    ent_np = convert_to_entropy_image(data)
    svd_tensor = preprocess(svd_np)
    ent_tensor = preprocess(ent_np)

    probs = predict(svd_tensor, ent_tensor)
    pred_idx = np.argmax(probs)
    print(f"\n예측 결과: {CLASS_NAMES[pred_idx]} ({probs[pred_idx]*100:.2f}%)")
    print(f"   → 전체 확률: {[f'{CLASS_NAMES[i]}: {p*100:.2f}%' for i, p in enumerate(probs)]}")

# 실행
if __name__ == "__main__":
    run_gui()
