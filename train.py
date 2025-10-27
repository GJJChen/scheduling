# -*- coding: utf-8 -*-
import argparse, os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # noqa: F401

from config import CFG
from dataset import SchedulingNPZDataset, split_dataset
from model import build_model
from rules import schedule_one

def set_seed(seed: int = 2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, optim, scheduler, device, epoch=None, is_hierarchical=False):
    model.train()
    crit = nn.CrossEntropyLoss()
    losses = []
    all_pred, all_tgt = [], []
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train Ep{epoch}" if epoch is not None else "Train")
    for feats, label in pbar:
        feats = feats.to(device)          # [B, 128, 6]
        label = label.to(device)          # [B]
        
        # 检查输入是否有 NaN
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print(f"Warning: NaN or Inf in input features")
            continue
        
        optim.zero_grad()
        
        if is_hierarchical:
            # HierTransformer: 返回 (service_logits, user_logits)
            service_logits, user_logits = model(feats)  # [B, 3], [B, 128]
            
            # 检查输出是否有 NaN
            if torch.isnan(service_logits).any() or torch.isinf(service_logits).any() or \
               torch.isnan(user_logits).any() or torch.isinf(user_logits).any():
                print(f"Warning: NaN or Inf in logits")
                continue
            
            # 从 label 中解析 service 和 user
            # label 格式: user * 3 + service (combined mode)
            service_tgt = label % 3
            user_tgt = label // 3
            
            # 分层损失
            loss_service = crit(service_logits, service_tgt)
            loss_user = crit(user_logits, user_tgt)
            loss = loss_service + loss_user
            
            # 预测：先预测 service，再预测该 service 下的 user
            pred_service = torch.argmax(service_logits, dim=1)
            pred_user = torch.argmax(user_logits, dim=1)
            pred = pred_user * 3 + pred_service  # 组合预测
            
        else:
            # 普通模型：单一输出
            logits = model(feats)             # [B, num_classes]
            
            # 检查输出是否有 NaN
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN or Inf in logits")
                continue
            
            loss = crit(logits, label)
            pred = torch.argmax(logits, dim=1)
        
        # 检查 loss 是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected, skipping batch")
            continue
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        losses.append(loss.item())
        all_pred.append(pred.detach().cpu().numpy())
        all_tgt.append(label.detach().cpu().numpy())

        # running accuracy for progress bar
        correct += (pred == label).sum().item()
        total += label.size(0)
        running_acc = correct / total if total > 0 else 0.0
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}")
    
    if scheduler is not None:
        scheduler.step()

    all_pred = np.concatenate(all_pred)
    all_tgt = np.concatenate(all_tgt)
    acc = accuracy_score(all_tgt, all_pred)
    return float(np.mean(losses)) if losses else float('inf'), float(acc)

@torch.no_grad()
def evaluate(model, loader, device, epoch=None, is_hierarchical=False):
    model.eval()
    crit = nn.CrossEntropyLoss()
    losses = []
    all_pred, all_tgt = [], []
    all_logits = []
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Val Ep{epoch}" if epoch is not None else "Val", leave=False)
    for feats, label in pbar:
        feats = feats.to(device)
        label = label.to(device)
        
        if is_hierarchical:
            # HierTransformer: 返回 (service_logits, user_logits)
            service_logits, user_logits = model(feats)  # [B, 3], [B, 128]
            
            # 从 label 中解析 service 和 user
            service_tgt = label % 3
            user_tgt = label // 3
            
            # 分层损失
            loss_service = crit(service_logits, service_tgt)
            loss_user = crit(user_logits, user_tgt)
            loss = loss_service + loss_user
            
            # 预测：先预测 service，再预测该 service 下的 user
            pred_service = torch.argmax(service_logits, dim=1)
            pred_user = torch.argmax(user_logits, dim=1)
            pred = pred_user * 3 + pred_service  # 组合预测
            
            # 构建组合 logits 用于 top-k 计算
            # 对于每个 service，使用对应的 user_logits
            B = feats.size(0)
            combined_logits = torch.zeros(B, 384, device=device)  # [B, 128*3]
            for s in range(3):
                combined_logits[:, s::3] = user_logits.unsqueeze(-1).expand(-1, -1, 1).squeeze(-1) + \
                                           service_logits[:, s:s+1] * 0.1  # 添加 service 信息
            logits = combined_logits
            
        else:
            # 普通模型：单一输出
            logits = model(feats)
            loss = crit(logits, label)
            pred = torch.argmax(logits, dim=1)
        
        losses.append(loss.item())
        all_pred.append(pred.detach().cpu().numpy())
        all_tgt.append(label.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

        correct += (pred == label).sum().item()
        total += label.size(0)
        running_acc = correct / total if total > 0 else 0.0
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}")
    all_pred = np.concatenate(all_pred)
    all_tgt = np.concatenate(all_tgt)
    all_logits = np.concatenate(all_logits)
    
    # Top-k 准确率计算
    top5_preds = np.argsort(all_logits, axis=1)[:, -5:]
    top10_preds = np.argsort(all_logits, axis=1)[:, -10:]
    top5_acc = np.mean([all_tgt[i] in top5_preds[i] for i in range(len(all_tgt))])
    top10_acc = np.mean([all_tgt[i] in top10_preds[i] for i in range(len(all_tgt))])
    
    acc = accuracy_score(all_tgt, all_pred)
    return float(np.mean(losses)), float(acc), float(top5_acc), float(top10_acc)


def check_dataset_labels(npz_path: str, sample_limit: Optional[int] = None) -> float:
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.int64)

    total = X.shape[0]
    if sample_limit is not None:
        total = min(total, sample_limit)

    meta = data["meta"].item() if "meta" in data else {}
    label_mode = meta.get("label_mode", "user")
    num_services = len(meta.get("services", CFG.SERVICES))

    correct = 0
    missing = 0
    mismatches = []

    for idx in range(total):
        snap = X[idx]
        u, s, _ = schedule_one(snap)

        if u is None or s is None:
            missing += 1
            continue

        if label_mode == "combined":
            rule_label = u * num_services + s
        else:
            rule_label = u

        dataset_label = int(y[idx])
        if rule_label == dataset_label:
            correct += 1
        elif len(mismatches) < 5:
            mismatches.append((idx, rule_label, dataset_label, int(u), int(s)))

    effective = total - missing
    accuracy = correct / effective if effective > 0 else 0.0

    print("=== 规则标签一致性检查 ===")
    print(f"检查样本: {total}")
    if missing:
        print(f"跳过样本: {missing} (schedule_one 返回 None)")
    print(f"标签准确率: {accuracy:.4f}")
    if mismatches:
        print("注意: 发现标签与规则不一致的样本 (最多5条):")
        for idx, rule_label, dataset_label, user_idx, service_idx in mismatches:
            print(f"  idx={idx} rule={rule_label} dataset={dataset_label} (user={user_idx}, service={service_idx})")

    return accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/train.npz", help="npz 数据集路径")
    ap.add_argument("--model", type=str, default="hier", choices=["bilstm", "mlp", "transformer", "hier"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)  # 降低batch size提升稳定性
    ap.add_argument("--lr", type=float, default=1e-2)  # 进一步降低学习率
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--save", type=str, default="checkpoints/best.pt")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (Windows建议0)")
    ap.add_argument("--overfit-test", action="store_true", help="小数据集过拟合测试模式")
    ap.add_argument("--label-check-samples", type=int, default=None,
                    help="预先校验标签时最多检查的样本数（默认全部）")
    ap.add_argument("--skip-label-check", action="store_true", help="跳过规则标签准确性校验")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.skip_label_check:
        check_dataset_labels(args.data, sample_limit=args.label_check_samples)

    # 数据
    full = SchedulingNPZDataset(args.data)
    num_classes = full.num_classes
    label_mode = full.label_mode
    print(f"数据集标签模式: {label_mode}, 类别数: {num_classes}")
    
    # 过拟合测试模式：只用前2048样本
    if args.overfit_test:
        print("*** 过拟合测试模式：使用前2048样本 ***")
        from torch.utils.data import Subset
        indices = list(range(min(2048, len(full))))
        full = Subset(full, indices)
        args.val_ratio = 0.2  # 测试模式用更多验证集
    
    train_ds, val_ds = split_dataset(full, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True if args.num_workers > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True if args.num_workers > 0 else False)

    # 判断是否使用分层模型
    is_hierarchical = (args.model == "hier")
    
    # 模型
    if is_hierarchical:
        from model import HierTransformer
        model = HierTransformer(input_dim=6).to(device)
        print(f"模型: HierTransformer (分层输出), 输入维度: 6")
    else:
        model = build_model(args.model, input_dim=6, num_classes=num_classes).to(device)
        print(f"模型: {args.model}, 输入维度: 6, 输出类别数: {num_classes}")
    
    # 初始化模型权重
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # 小初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param, gain=0.1)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    model.apply(init_weights)
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr*0.01)

    best_acc = -1.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, scheduler, device, 
                                          epoch=epoch, is_hierarchical=is_hierarchical)
        va_loss, va_acc, va_top5, va_top10 = evaluate(model, val_loader, device, 
                                                       epoch=epoch, is_hierarchical=is_hierarchical)
        current_lr = optim.param_groups[0]['lr']
        tqdm.write(f"[Epoch {epoch:02d}] Train loss={tr_loss:.4f} acc={tr_acc:.4f} | Val loss={va_loss:.4f} acc={va_acc:.4f} top5={va_top5:.4f} top10={va_top10:.4f} | LR={current_lr:.6f}")
        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(dict(
                model=args.model, state_dict=model.state_dict(),
                cfg=dict(N_USERS=CFG.N_USERS, num_classes=num_classes, label_mode=label_mode)
            ), args.save)
            tqdm.write(f"  -> 保存最佳模型到: {args.save} (val_acc={best_acc:.4f})")

    print(f"训练完成。最佳验证准确率 Top-1={best_acc:.4f}")

if __name__ == "__main__":
    main()
