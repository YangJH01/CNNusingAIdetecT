import torch
import torch.nn as nn
import timm

def model_feature_extractor():
    model = timm.create_model('convnext_tiny.fb_in22k', pretrained=True, num_classes=0)
    # ConvNeXt의 경우 head 제거하여 feature만 추출
    model.head = nn.Identity()
    return model

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

class DualBranchModel(nn.Module):
    def __init__(self, base_cnn, num_classes=2):
        super().__init__()

        # Backbone
        self.svd_backbone = base_cnn()
        self.ent_backbone = base_cnn()

        # Feature 차원 추정
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.svd_backbone(dummy)
            self.need_pooling = len(out.shape) == 4
            self.feat_dim = out.shape[1]

        # SEBlock(attention)
        self.attn_svd = SEBlock(self.feat_dim)
        self.attn_ent = SEBlock(self.feat_dim)

        # Pooling
        self.pool = nn.AdaptiveAvgPool2d(1) if self.need_pooling else nn.Identity()

        # backbone dropout
        dropout_backbone = 0.1  # Default dropout value
        self.backbone_dropout = nn.Dropout(dropout_backbone) if dropout_backbone > 0 else nn.Identity()

        # Fusion(단순 concat 대신 사용함)
        self.fusion = CrossAttentionFusion(self.feat_dim)

        # Classifier(분류, 과적합 피하기 위해 최대한 간단하게 설정)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.5),
            nn.Linear(self.feat_dim, num_classes)
        )

    def forward(self, x_svd, x_ent):
        svd_feat = self.svd_backbone(x_svd)
        ent_feat = self.ent_backbone(x_ent)

        # Attention
        svd_feat = self.attn_svd(svd_feat)
        ent_feat = self.attn_ent(ent_feat)

        # Dropout after SE
        svd_feat = self.backbone_dropout(svd_feat)
        ent_feat = self.backbone_dropout(ent_feat)

        # Pooling
        if self.need_pooling:
            svd_feat = self.pool(svd_feat).squeeze(-1).squeeze(-1)
            ent_feat = self.pool(ent_feat).squeeze(-1).squeeze(-1)

        # Fusion
        fused = self.fusion(svd_feat, ent_feat)

        return self.classifier(fused)
