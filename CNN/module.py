'''
dual_branch_ablation_runner.py를 위한 함수들 모아둔 것
'''



import os
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import random

# ========== 환경 설정 ==========
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# ========== Early Stopping 클래스 ==========
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', monitor='acc'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.monitor = monitor  # 'loss' or 'acc'
        
        if monitor == 'loss':
            self.best_score = np.inf
            self.is_better = lambda score, best: score < best - delta
        else:  # accuracy
            self.best_score = -np.inf
            self.is_better = lambda score, best: score > best + delta
        
    def __call__(self, val_metric, model):
        if self.is_better(val_metric, self.best_score):
            self.best_score = val_metric
            self.save_checkpoint(val_metric, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            print(f'Validation {self.monitor} improved to {val_metric:.6f}. 모델 저장...')
        torch.save(model.state_dict(), self.path)

# ========== 데이터 전처리 함수들 ==========
def get_svd_train_transform():
    """SVD 이미지용 증강 - 색상과 구조 정보 보존 중심"""
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=8, p=0.3),  # 작은 회전
        A.Affine(
            scale=(0.98, 1.02),  # 작은 스케일 변화
            translate_percent=(0.02, 0.02),  # 작은 이동
            p=0.3
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, 
            contrast_limit=0.15, 
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=8, 
            sat_shift_limit=12, 
            val_shift_limit=8, 
            p=0.3
        ),
        A.CoarseDropout(
            num_holes_range=(2, 2),
            hole_height_range=(16, 16),
            hole_width_range=(16, 16),
            p=0.2
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
def get_entropy_train_transform():
    """Entropy 이미지용 증강 - 텍스처 정보 강화 중심"""
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, p=0.4),  # 더 큰 회전 허용
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.4),  # 대비 강화
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),  # 엣지 강화
        A.GaussNoise(
            std_range=(0.05, 0.15),
            mean_range=(0, 0),
            per_channel=True,
            noise_scale_factor=1.0,
            p=0.2
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.2,  # 대비 변화 더 허용
            p=0.3
        ),
        A.GridDropout(ratio=0.15, p=0.25),  # 패턴 일부 제거
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform():
    """검증/테스트용 - 증강 없음"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ========== 데이터셋 클래스 ==========
class DualImageDataset(Dataset):
    def __init__(self, svd_dir, ent_dir, svd_transform=None, ent_transform=None):
        self.svd_transform = svd_transform
        self.ent_transform = ent_transform
        self.classes = os.listdir(svd_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.targets = []
        
        # 파일명을 기준으로 매핑 생성
        self.samples = []
        for cls in self.classes:
            svd_class_dir = os.path.join(svd_dir, cls)
            ent_class_dir = os.path.join(ent_dir, cls)
            
            svd_files = set(os.listdir(svd_class_dir))
            ent_files = set(os.listdir(ent_class_dir))
            common_files = sorted(svd_files.intersection(ent_files))
            
            class_idx = self.class_to_idx[cls]
            for file in common_files:
                svd_path = os.path.join(svd_class_dir, file)
                ent_path = os.path.join(ent_class_dir, file)
                self.samples.append((svd_path, ent_path, class_idx))
                self.targets.append(class_idx)
    
    def __getitem__(self, idx):
        svd_path, ent_path, label = self.samples[idx]
        
        svd_img = cv2.imread(svd_path)
        ent_img = cv2.imread(ent_path)
        
        svd_img = cv2.cvtColor(svd_img, cv2.COLOR_BGR2RGB)
        ent_img = cv2.cvtColor(ent_img, cv2.COLOR_BGR2RGB)

        # 각각 다른 증강 적용
        if self.svd_transform:
            svd_img = self.svd_transform(image=svd_img)["image"]
        
        if self.ent_transform:
            ent_img = self.ent_transform(image=ent_img)["image"]
        
        return svd_img, ent_img, label
    
    def __len__(self):
        return len(self.samples)

# ========== Attention 모듈들 ==========
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, svd_feat, ent_feat):
        # 입력이 이미 1D vector여야 함 [B, D]
        if len(svd_feat.shape) != 2 or len(ent_feat.shape) != 2:
            raise ValueError(f"Expected 2D tensors, got svd: {svd_feat.shape}, ent: {ent_feat.shape}")
            
        # [B, D] → [B, 1, D]
        svd = svd_feat.unsqueeze(1)
        ent = ent_feat.unsqueeze(1)

        # stack → [B, 2, D]
        x = torch.cat([svd, ent], dim=1)

        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.mlp(x)

        # fusion 결과: 평균
        fused = x.mean(dim=1)
        return fused

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ========== 모델 클래스들 ==========
class DualBranchCNN(nn.Module):
    def __init__(self, base_cnn, num_classes: int = 2, dropout_backbone: float = 0.0, dropout_fc: float = 0.5):
        super().__init__()
        self.svd_branch = base_cnn()
        self.ent_branch = base_cnn()
        
        # optional backbone-dropout
        self.backbone_drop = (
            nn.Dropout(dropout_backbone) if dropout_backbone > 0.0 else nn.Identity()
        )
        
        print(f"Backbone dropout: {dropout_backbone}, FC dropout: {dropout_fc}")
        
        # 특징 벡터 차원 자동 감지
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feat_dim = self.svd_branch(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(feat_dim*2, 256),  # 자동 계산된 차원 사용
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(256, num_classes)
        )

    def forward(self, svd_img, ent_img):
        svd_feat = self.backbone_drop(self.svd_branch(svd_img))
        ent_feat = self.backbone_drop(self.ent_branch(ent_img))
        x = torch.cat([svd_feat, ent_feat], dim=1)
        return self.fc(x)

class DualBranchAttentionModel(nn.Module):
    def __init__(self, base_cnn, num_classes=2,
                 dropout_backbone=0.0, dropout_fc=0.5,
                 attn_type='default', use_attention=True):
        super().__init__()

        # Base model을 수정하여 feature extractor로 사용
        self.svd_backbone = base_cnn()
        self.ent_backbone = base_cnn()
        
        self.attn_type = attn_type.lower()
        self.use_attention = use_attention

        # Feature dimension 측정
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_feat = self.svd_backbone(dummy_input)
            print(f"Backbone output shape: {dummy_feat.shape}")
            
            if len(dummy_feat.shape) == 4:  # [B, D, H, W]
                feat_dim = dummy_feat.shape[1]
                self.need_pooling = True
                print(f"Feature map detected: {dummy_feat.shape}, will apply pooling")
            elif len(dummy_feat.shape) == 2:  # [B, D]
                feat_dim = dummy_feat.shape[1]
                self.need_pooling = False
                print(f"Feature vector detected: {dummy_feat.shape}, no pooling needed")
            else:
                raise ValueError(f"Unexpected backbone output shape: {dummy_feat.shape}")

        # Attention 모듈들 (2D feature map용만)
        if self.attn_type == 'cbam' and self.need_pooling:
            self.attn_svd = CBAM(feat_dim)
            self.attn_ent = CBAM(feat_dim)
            print(f"Using CBAM attention with {feat_dim} channels")
        elif self.attn_type == 'se' and self.need_pooling:
            self.attn_svd = SEBlock(feat_dim)
            self.attn_ent = SEBlock(feat_dim)
            print(f"Using SE attention with {feat_dim} channels")
        else:
            self.attn_svd = nn.Identity()
            self.attn_ent = nn.Identity()
            print("No spatial attention applied")

        self.backbone_drop = nn.Dropout(dropout_backbone) if dropout_backbone > 0.0 else nn.Identity()

        # Pooling layer (필요시)
        self.pool = nn.AdaptiveAvgPool2d(1) if self.need_pooling else nn.Identity()

        # Cross-attention fusion
        if self.use_attention:
            self.fusion = CrossAttentionFusion(feat_dim)
            print(f"Using CrossAttentionFusion with dim={feat_dim}")
        else:
            self.fusion = None
            print("Using simple concatenation fusion")

        # Classifier
        fc_input_dim = feat_dim if self.use_attention else feat_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(fc_input_dim),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_input_dim, num_classes)
        )
        print(f"Classifier input dim: {fc_input_dim}")

    def forward(self, x_svd, x_ent):
        # Backbone feature extraction
        svd_feat = self.svd_backbone(x_svd)  # [B, D] or [B, D, H, W]
        ent_feat = self.ent_backbone(x_ent)

        # Attention 적용 (2D feature map인 경우에만)
        if self.need_pooling:
            svd_feat = self.attn_svd(svd_feat)
            ent_feat = self.attn_ent(ent_feat)
            
        # Dropout 적용
        svd_feat = self.backbone_drop(svd_feat)
        ent_feat = self.backbone_drop(ent_feat)

        # Feature pooling (필요시)
        if self.need_pooling:
            svd_feat = self.pool(svd_feat).squeeze(-1).squeeze(-1)
            ent_feat = self.pool(ent_feat).squeeze(-1).squeeze(-1)

        # Fusion
        if self.use_attention:
            fused = self.fusion(svd_feat, ent_feat)
        else:
            fused = torch.cat([svd_feat, ent_feat], dim=1)

        return self.classifier(fused)

# ========== 모델 생성 함수 ==========
def model_feature_extractor():
    """ConvNeXt-Tiny 기반 feature extractor 생성"""
    model = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, num_classes=0)
    # ConvNeXt의 경우 head 제거하여 feature만 추출
    model.head = nn.Identity()
    return model

# ========== 유틸리티 함수들 ==========
def get_loader(svd_dir, ent_dir, svd_transform=None, ent_transform=None, batch_size=32, shuffle=False):
    """DataLoader 생성 함수"""
    dataset = DualImageDataset(svd_dir, ent_dir, svd_transform, ent_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    return loader, dataset

def get_dirs(base):
    """디렉토리 구조 생성 함수"""
    return {
        'train': os.path.join(base, "train"),
        'val': os.path.join(base, "val"),
        'test': os.path.join(base, "test"),
    }