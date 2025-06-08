"""
Input Folder 구조는
DATA
 ├── benign
 │      ├── file1
 │      ├── file2.vir
 │      ├── file3.bytes
 │      └── ...
 └── malware
        ├── file4
        ├── file5.vir
        ├── file6.bytes
        └── ...

와 같이 구성되어있다고 가정합니다.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from skimage.filters.rank import entropy
from skimage.morphology import disk
from PIL import ImageOps

# ================== 설정 ==================
INPUT_ROOT  = r'C:\workspace\SVD\datasets\every'
OUTPUT_ROOT = r'C:\workspace\SVD\new_data\every_entropy'
RESIZE_SIZE = (224, 224)
VALID_EXT = {'.bytes', '.vir'}   # 명시적 확장자
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# =========================================

# ---------- 유틸 ----------
def auto_width(length: int) -> int:
    """byte 길이에 따라 정사각형에 가장 가까운 width 반환."""
    cand = [4096, 3584, 3072, 2560, 2048, 1792, 1536, 1280,
            1024, 896, 768, 640, 512, 448, 384, 320,
             256, 224, 192, 160, 128, 96, 64, 48, 32]
    best, diff = None, float('inf')
    for w in cand:
        h = int(np.ceil(length / w))
        if h / w <= 2.0 and h <= 2048 and abs(h - w) < diff:
            best, diff = w, abs(h - w)
    return best or 4096

def read_bytes_file(path: str) -> list[int]:
    out = []
    with open(path, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            if len(line) < 9:
                continue
            for val in line[9:].strip().split():
                if val == '??':
                    out.append(0)
                elif len(val) == 2 and all(c in '0123456789abcdefABCDEF' for c in val):
                    try:
                        out.append(int(val, 16))
                    except ValueError:
                        out.append(0)
                else:
                    out.append(0)
    return out

def read_binary_file(path: str) -> list[int]:
    with open(path, 'rb') as f:
        return list(f.read())

def bytes_to_entropy(byte_array: list[int]) -> Image.Image:
    ln = len(byte_array)
    w  = auto_width(ln)
    h  = int(np.ceil(ln / w))
    padded = byte_array + [0] * (w * h - ln)
    mat = np.array(padded, dtype=np.uint8).reshape(h, w)
    
    # entropy 계산 (local neighborhood radius=5)
    ent = entropy(mat, disk(5))  # uint8 → float32
    ent = (255 * (ent / np.max(ent))).astype(np.uint8)  # 정규화 to 0~255
    ent_img = Image.fromarray(ent, mode='L').resize(RESIZE_SIZE, Image.BILINEAR) # 그레이스케일로 변환 후 리사이즈
    return ent_img
# ------------------------------------------

def process_one(args):
    src, dst, ext = args
    try:
        data = read_bytes_file(src) if ext == '.bytes' else read_binary_file(src)
        img  = bytes_to_entropy(data)
        img.save(dst, format='PNG')
        return True
    except Exception as e:
        with open('grayscale_errors.log', 'a') as logf:
            logf.write(f'{src} 실패: {e}\n')
        return False

def convert_all(input_root: str, output_root: str):
    tasks = []
    for sub in os.listdir(input_root):
        in_dir  = os.path.join(input_root, sub)
        if not os.path.isdir(in_dir):
            continue
        out_dir = os.path.join(output_root, sub)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(in_dir):
            src = os.path.join(in_dir, fname)
            if not os.path.isfile(src):
                continue

            base, ext = os.path.splitext(fname)
            ext = ext.lower()

            # 확장자 체크: .bytes / .vir / (없음 → 바이너리)
            if ext in VALID_EXT:
                use_ext = ext
            elif ext == '':
                use_ext = '.bin'     # 확장자 없는 파일 → 바이너리 처리
            else:
                continue             # 그 외 확장자 스킵

            dst = os.path.join(out_dir, base + '.png') #png 확장자로 저장
            tasks.append((src, dst, use_ext))

    print(f'총 파일 수: {len(tasks)}')
    with Pool(processes=cpu_count() // 2) as pool:   # 물리 코어만 사용
        ok = list(tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)))

    print(f'\n✅ 완료: {sum(ok)}개, ❌ 실패: {len(ok) - sum(ok)}개')

if __name__ == '__main__':
    convert_all(INPUT_ROOT, OUTPUT_ROOT)
