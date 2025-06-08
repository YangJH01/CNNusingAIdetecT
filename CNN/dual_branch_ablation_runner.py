
"""
Dual‑Branch ConvNeXt Ablation Runner
====================================
 - 사용법 (예시)
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
  
위와 같은 방식으로 각각 인자를 지정해서 사용함
"""

import argparse, os, math, random, time, copy, warnings
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import transforms
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import timm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ======== 데이터셋 & 모델 정의 ========
# 기존 different_arg_origin.py에서 그대로 가져옴 (경로만 수정)
from module import (
    DualImageDataset,
    get_svd_train_transform, get_entropy_train_transform, get_val_transform,
    DualBranchCNN, model_feature_extractor, EarlyStopping, DualBranchAttentionModel
)

def set_seed(seed:int = 42):
    import random, os, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- MixUp 구현 ----------

def mixup_data(x1, x2, y, alpha):
    if alpha <= 0.0:
        return x1, x2, y, None, None, 1.0  # no mixup
    lam = np.random.beta(alpha, alpha)
    batch_size = y.size(0)
    index = torch.randperm(batch_size).to(y.device)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, None, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---------- Adaptive Unfreeze ----------

def set_stage_trainable(convnext_model, stage_level):
    """stage_level: 0=FC only, 1=+stage4, 2=+stage3, 3=all"""
    for p in convnext_model.parameters():
        p.requires_grad = False
    if stage_level >= 1:
        for p in convnext_model.stages[3].parameters():
            p.requires_grad = True
    if stage_level >= 2:
        for p in convnext_model.stages[2].parameters():
            p.requires_grad = True
    if stage_level >= 3:
        for p in convnext_model.parameters():
            p.requires_grad = True

# ---------- Argument Parser ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, default='debug')
    p.add_argument('--attention', action='store_true')
    p.add_argument('--attn_type', type=str, default='default', choices=['default', 'cbam', 'se'],
                   help='Attention type: default, cbam, or se')
    p.add_argument('--base_from', type=str, default=None,
                   help='previous exp_name to load best_model.pth from')
    p.add_argument('--svd_dir', type=str, default=r'C:\workspace\SVD\new_data\new_every_color_single_split_same')
    p.add_argument('--ent_dir', type=str, default=r'C:\workspace\SVD\new_data\new_every_entropy_split_same')
    p.add_argument('exp_dir', type=str, nargs='?', default=r'C:\workspace\SVD\result')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--label_smoothing', type=float, default=0.02)
    p.add_argument('--mixup', type=float, default=0.0, help='alpha value; 0 disables')
    p.add_argument('--dropout_fc', type=float, default=0.5)
    p.add_argument('--dropout_backbone', type=float, default=0.0)
    p.add_argument('--freeze_scheme', type=str, default='0:0,10:1,20:2',
                   help='comma‑sep list epoch:stage_level')
    p.add_argument("--warmup", type=int, default=0, help="warm‑up epochs after unfreeze")
    p.add_argument('--ema', action='store_true')
    p.add_argument('--swa', type=int, default=0,
               help='0=off, N>0 → 마지막 N epoch 동안 SWA')
    p.add_argument('--batch_mode', action='store_true', help='run EXPERIMENTS dict')
    p.add_argument(
        '--early_stopping',
        nargs='?',
        const=7,
        type=int,
        help='Enable EarlyStopping (default patience=7 if no value is given)'
    )
    p.add_argument('--opt_type', type=str, default='adamw',
                   choices=['adamw', 'adabelief', 'lookahead_adamw', 'lookahead_adabelief'],
                   help='Optimizer type: adamw, adabelief, lookahead_adamw, lookahead_adabelief')
    return p.parse_args()

# ---------- Training Utilities ----------
def make_optimizer(model, base_lr, weight_decay, opt_type="adamw"):
    classifier_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {'params': classifier_params, 'lr': base_lr},
        {'params': backbone_params, 'lr': base_lr * 0.1},
    ]

    if opt_type == "adamw":
        return optim.AdamW(param_groups, weight_decay=weight_decay)

    elif opt_type == "adabelief":
        from torch_optimizer import AdaBelief
        return AdaBelief(
            param_groups,
            lr=base_lr,  # 이건 param_group별로 무시됨
            eps=1e-16,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            rectify=True  # 안정화
        )

    elif opt_type == "lookahead_adamw":
        from torch_optimizer import Lookahead
        base_opt = optim.AdamW(param_groups, weight_decay=weight_decay)
        return Lookahead(base_opt)

    elif opt_type == "lookahead_adabelief":
        from torch_optimizer import AdaBelief
        from torch_optimizer import Lookahead
        base_opt = AdaBelief(
            param_groups,
            lr=base_lr,  # 이건 param_group별로 무시됨
            eps=1e-16,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            rectify=True  # 안정화
        )
        return Lookahead(base_opt)

    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    
def load_checkpoint_if_any(model, cfg):
    if not cfg['base_from']:
        return 0.0
    ckpt_path = os.path.join(cfg['exp_dir'], cfg['base_from'], 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f"[WARN] base_from checkpoint not found: {ckpt_path}")
        return 0.0
    try:
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
        print(f"Warm‑start from {ckpt_path}")
        return 1.0  # dummy val acc (unknown)
    except Exception as e:
        print(f"[WARN] failed to load checkpoint: {e}")
        return 0.0

# ---------- Main Training Loop ----------

def run_experiment(cfg):
    set_seed(42) # seed 고정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(cfg['exp_dir'], cfg['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)

    # dataset / dataloader
    train_set = DualImageDataset(
        os.path.join(cfg['svd_dir'], 'train'),
        os.path.join(cfg['ent_dir'], 'train'),
        get_svd_train_transform(), get_entropy_train_transform())
    val_set   = DualImageDataset(
        os.path.join(cfg['svd_dir'], 'val'),
        os.path.join(cfg['ent_dir'], 'val'),
        get_val_transform(), get_val_transform())
    test_set  = DualImageDataset(
        os.path.join(cfg['svd_dir'], 'test'),
        os.path.join(cfg['ent_dir'], 'test'),
        get_val_transform(), get_val_transform())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=cfg['batch_size'])
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=cfg['batch_size'])

    # model
    base_cnn = model_feature_extractor
    if cfg['attention']:
        model = DualBranchAttentionModel(
            base_cnn, num_classes=2, 
            dropout_fc=cfg['dropout_fc'], 
            dropout_backbone=cfg['dropout_backbone'],
            attn_type=cfg.get('attn_type', 'default'),
            use_attention=True
        ).to(device)
        print(f"Using DualBranchAttentionModel with attention type: {cfg.get('attn_type', 'default')}")
    else:
        model = DualBranchAttentionModel(
            base_cnn, num_classes=2, 
            dropout_fc=cfg['dropout_fc'], 
            dropout_backbone=cfg['dropout_backbone'],
            attn_type=cfg.get('attn_type', 'default'),
            use_attention=False
        ).to(device)
        print(f"Using DualBranchAttentionModel with attention type: {cfg.get('attn_type', 'default')}")
    
    # 모델 forward 테스트
    model.eval()
    with torch.no_grad():
        try:
            dummy_svd = torch.randn(2, 3, 224, 224).to(device)
            dummy_ent = torch.randn(2, 3, 224, 224).to(device)
            dummy_out = model(dummy_svd, dummy_ent)
            print(f"✅ Model forward test passed. Output shape: {dummy_out.shape}")
        except Exception as e:
            print(f"❌ Model forward test failed: {e}")
            raise e
        
    load_checkpoint_if_any(model, cfg)
    
    # criterion
    class_weights = compute_class_weight('balanced', classes=np.unique(train_set.targets), y=train_set.targets)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
                                    label_smoothing=cfg['label_smoothing'])

    # opt / sched
    optimizer = make_optimizer(model, cfg['lr'], cfg['weight_decay'], cfg['opt_type'])
    
    #원본
    if cfg['opt_type'] == 'adamw' or cfg['opt_type'] == 'adabelief':
        print(f"{cfg['opt_type']} / using CosineAnnealingWarmRestarts scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=3,  # 주기(최초 epoch)
            T_mult=2, # 주기 증가 배수 ex) 3, 6, 12, 24 (지수적으로 증가)
            eta_min=1e-6 # 마지노선
        )
    
    elif cfg['opt_type'] == 'lookahead_adamw' or cfg['opt_type'] == 'lookahead_adabelief':
        # cosine annealing lr - with lookahead
        print(f"{cfg['opt_type']} / using CosineAnnealingLR with Lookahead optimizer")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'],     # 총 epoch 수
            eta_min=1e-6
        )

    # EMA / SWA
    ema_model = AveragedModel(model) if cfg['ema'] else None
    swa_model, swa_scheduler = None, None
    if cfg['swa']:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg['lr']*0.1)

    # freeze schedule parsing
    freeze_points = {int(k): int(v) for k, v in (s.split(":") for s in cfg["freeze_scheme"].split(","))}
    current_stage_level = None
    warmup_left = 0
    warmup_offset = None
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    
    # early stopping
    early_stopper = None
    if cfg.get('early_stopping') is not None:
        early_stopper = EarlyStopping(
            patience=cfg['early_stopping'],
            verbose=True,
            delta=0.001,
            path=os.path.join(exp_dir, 'best_model.pth'),
            monitor='acc'  # accuracy 기준으로 모니터링
        )

    best_val = 0.0
    print(f"early_stopping: {early_stopper is not None}, patience={cfg.get('early_stopping', 'N/A')}")
    
    for epoch in range(cfg['epochs']):
        # adaptive unfreeze
        if epoch in freeze_points:
            current_stage_level = freeze_points[epoch]
            # ConvNeXt 구조에 맞게 수정
            if hasattr(model, 'svd_backbone') and hasattr(model.svd_backbone, 'stages'):
                set_stage_trainable(model.svd_backbone, current_stage_level)
                set_stage_trainable(model.ent_backbone, current_stage_level)
            else:
                print("Warning: Model doesn't have expected ConvNeXt structure for unfreeze")

            warmup_left = cfg["warmup"]
            warmup_offset = epoch
            base_lrs = [g["lr"] for g in optimizer.param_groups]
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

            print(f"[Epoch {epoch}] Unfreeze to stage level {current_stage_level}, warm-up {cfg['warmup']} epochs")

        model.train()
        running_loss, correct = 0.0, 0
        for svd, ent, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]}'):
            svd, ent, label = svd.to(device), ent.to(device), label.to(device)
            optimizer.zero_grad()

            # MixUp handling
            if cfg['mixup'] > 0.0:
                svd, ent, _, y_a, y_b, lam = mixup_data(svd, ent, label, cfg['mixup'])
                out = model(svd, ent)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                out = model(svd, ent)
                loss = criterion(out, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * label.size(0)
            correct += (out.argmax(1) == label).sum().item()

            if ema_model:  # EMA update per batch
                ema_model.update_parameters(model)

        train_loss = running_loss / len(train_set)
        train_acc  = correct / len(train_set)

        # ------------ validation ------------
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for svd, ent, label in val_loader:
                svd, ent, label = svd.to(device), ent.to(device), label.to(device)
                out = model(svd, ent)
                loss = criterion(out, label)
                v_loss += loss.item() * label.size(0)
                v_correct += (out.argmax(1) == label).sum().item()
        val_loss = v_loss / len(val_set)
        val_acc  = v_correct / len(val_set)

        # Learning rate scheduling
        if warmup_left > 0:
            warmup_left -= 1
            ratio = 1.0 - warmup_left / max(1, cfg["warmup"])
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * (0.1 + 0.9 * ratio)
            print(f"Warmup epoch {cfg['warmup'] - warmup_left}/{cfg['warmup']}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        elif warmup_left == 0 and warmup_offset is not None:
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base
            warmup_offset = None
            scheduler.step()    # cosineannealingwarmrestarts
            #scheduler.step(val_acc) # reduceLROnPlateau
        else:
            scheduler.step()
            #scheduler.step(val_acc)
            
        if cfg['swa'] and epoch >= cfg['epochs'] - cfg['swa']:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        print(f"Epoch {epoch+1}: TrainAcc={train_acc:.4f} TrainLoss={train_loss:.4f} | ValAcc={val_acc:.4f} ValLoss={val_loss:.4f}")
        print(f"[Epoch {epoch}] LR = {optimizer.param_groups[0]['lr']:.8f}")
        
        if early_stopper:
            early_stopper(val_acc, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break
        else: 
            if val_acc > best_val:
                print(f"🌈previous best val_acc: {best_val:.4f} → new best: {val_acc:.4f}")
                best_val = val_acc
                torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))

    # ------------ Evaluation (best model) ------------
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth')))
    if cfg['swa']:
        torch.save(swa_model.module.state_dict(), os.path.join(exp_dir, 'best_model_swa.pth'))
        model.load_state_dict(swa_model.module.state_dict())

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for svd, ent, label in tqdm(test_loader, desc='Testing'):
            svd, ent = svd.to(device), ent.to(device)
            out = model(svd, ent)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(label.numpy())
    report = classification_report(labels, preds, target_names=['benign','malware'], digits=4)
    with open(os.path.join(exp_dir, 'test_report.txt'), 'w') as f:
        f.write(report)
    print(f"BEST VALIDATION: {best_val}")
    print(report)

# ---------- batch mode examples ----------
EXPERIMENTS = {
    'baseline_no_mixup':   {'mixup':0.0, 'label_smoothing':0.02, 'dropout_fc':0.5},
    'mixup0.2_ls0.0':      {'mixup':0.2, 'label_smoothing':0.0,  'dropout_fc':0.5},
    'mixup0.2_ls0.0_dp07': {'mixup':0.2, 'label_smoothing':0.0,  'dropout_fc':0.7},
    'unfreeze_only':       {'mixup':0.0, 'label_smoothing':0.02, 'dropout_fc':0.5,
                            'freeze_scheme':'0:0,10:1,20:2'},
}

if __name__ == '__main__':
    args = parse_args()
    if not args.batch_mode:
        cfg = vars(args)
        cfg['exp_name'] = args.exp_name
        run_experiment(cfg)
    else:
        base_cfg = vars(args)
        for name, mod in EXPERIMENTS.items():
            cfg = copy.deepcopy(base_cfg)
            cfg.update(mod)
            cfg['exp_name'] = name
            print(f"\n=== Running {name} ===")
            run_experiment(cfg)
