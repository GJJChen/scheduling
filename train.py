# -*- coding: utf-8 -*-
import argparse, os, json
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import SchedulingNPZDataset, split_dataset
from model import build_model



class EarlyStopping:
    """早停"""
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: 忍耐多少个epoch性能不提升，超过之后停止训练
            min_delta: 最小改进阈值，改进小于该值不计入提升
            mode: “max” 表示指标越大越好（如accuracy），“min” 表示越小越好（如loss）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'max':
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class TrainingHistory:
    """记录训练历史"""
    def __init__(self, model_name, save_dir="results"):
        self.model_name = model_name
        self.save_dir = save_dir
        self.history = {
            'model': model_name,
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top5': [],
            'val_top10': [],
            'learning_rate': []
        }
        os.makedirs(save_dir, exist_ok=True)
        
    def add_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, val_top5, val_top10, lr):
        """添加一个epoch的记录"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['train_acc'].append(float(train_acc))
        self.history['val_loss'].append(float(val_loss))
        self.history['val_acc'].append(float(val_acc))
        self.history['val_top5'].append(float(val_top5))
        self.history['val_top10'].append(float(val_top10))
        self.history['learning_rate'].append(float(lr))
    
    def add_test_results(self, test_loss, test_acc, test_top5, test_top10):
        """添加测试结果"""
        self.history['test_loss'] = float(test_loss)
        self.history['test_acc'] = float(test_acc)
        self.history['test_top5'] = float(test_top5)
        self.history['test_top10'] = float(test_top10)
    
    def save(self):
        """存到JSON"""
        save_path = os.path.join(self.save_dir, f"{self.model_name}_history.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存到: {save_path}")
        return save_path


def set_seed(seed: int = 2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, optim, scheduler, device, epoch=None,
                    amp_enabled=False, scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    non_blocking: bool = False):
    model.train()
    crit = nn.CrossEntropyLoss()
    losses = []
    all_pred, all_tgt = [], []
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train Ep{epoch}" if epoch is not None else "Train")
    for feats, label in pbar:
        feats = feats.to(device, non_blocking=non_blocking)          # [B, 128, 6]
        label = label.to(device, non_blocking=non_blocking)          # [B]
        
        # 检查输入是否有 NaN 的值
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print(f"警告: 存在NaN 或 Inf")
            continue
        
        optim.zero_grad()
        
        # 前向传播过程
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            logits = model(feats)             # [B, num_classes]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"警告: 在输出中检测到 NaN 或 Inf")
                continue

            loss = crit(logits, label)
            pred = torch.argmax(logits, dim=1)
        
        # 检查 loss 是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告: 检测到NaN or Inf 损失")
            continue
        
        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

        losses.append(loss.item())
        all_pred.append(pred.detach().cpu().numpy())
        all_tgt.append(label.detach().cpu().numpy())

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
def evaluate(model, loader, device, epoch=None, amp_enabled=False,
             non_blocking: bool = False):
    model.eval()
    crit = nn.CrossEntropyLoss()
    losses = []
    all_pred, all_tgt = [], []
    all_logits = []
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Val Ep{epoch}" if epoch is not None else "Val", leave=False)
    for feats, label in pbar:
        feats = feats.to(device, non_blocking=non_blocking)
        label = label.to(device, non_blocking=non_blocking)
        
        with torch.amp.autocast('cuda', enabled=amp_enabled):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/train.npz", help="npz 训练数据集路径")
    ap.add_argument("--test-data", type=str, help="npz 测试数据集路径（可选，从不同场景加载测试集）")
    ap.add_argument("--model", type=str, default="bilstm", choices=["bilstm", "mlp", "transformer", "hier"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)  # 降低batch size提升稳定性
    ap.add_argument("--lr", type=float, default=1e-4)  # 进一步降低学习率
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--save", type=str, default="checkpoints/best.pt")
    ap.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (Windows建议0)")
    ap.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader 预取因子(>0且num_workers>0生效)")
    ap.add_argument("--persistent-workers", action="store_true", help="开启持久化workers以减少反复fork开销")
    # 早停相关参数
    ap.add_argument("--patience", type=int, default=15, help="早停忍耐多少个epoch不提升就停止")
    ap.add_argument("--min-delta", type=float, default=0.0001, help="早停最小改进阈值")
    ap.add_argument("--no-early-stop", action="store_true", help="禁用早停")
    # AMP/TF32 加速
    ap.add_argument("--amp", dest="amp", action="store_true", help="启用混合精度训练(默认CUDA上开启)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="禁用混合精度训练")
    ap.set_defaults(amp=None)
    ap.add_argument("--allow-tf32", dest="tf32", action="store_true", help="允许TF32(需Ampere+显卡)")
    ap.add_argument("--no-tf32", dest="tf32", action="store_false", help="禁用TF32")
    ap.set_defaults(tf32=None)
    # 训练历史保存
    ap.add_argument("--results-dir", type=str, default="results", help="训练历史保存目录")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 默认在CUDA上开启AMP/TF32
    if args.amp is None:
        args.amp = (device == "cuda")
    if args.tf32 is None:
        args.tf32 = (device == "cuda")
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        except Exception:
            pass
        # 对部分算子提升速度
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    # 数据
    if args.test_data:
        # 使用不同数据集作为训练和测试
        train_full = SchedulingNPZDataset(args.data, normalize=True, precompute=True)
        test_full = SchedulingNPZDataset(args.test_data, normalize=True, precompute=True)
        
        # 检查标签模式是否一致
        if train_full.label_mode != test_full.label_mode:
            raise ValueError(f"训练集和测试集标签模式不一致: {train_full.label_mode} vs {test_full.label_mode}")
        if train_full.num_classes != test_full.num_classes:
            raise ValueError(f"训练集和测试集类别数不一致: {train_full.num_classes} vs {test_full.num_classes}")
        
        # 从训练集中分割出验证集
        train_ds, val_ds = split_dataset(train_full, val_ratio=args.val_ratio, seed=args.seed)
        # 测试集作为独立的测试数据
        from torch.utils.data import DataLoader as DL
        test_loader = DL(test_full, batch_size=args.batch_size, shuffle=False, **dl_common)
        
        num_classes = train_full.num_classes
        label_mode = train_full.label_mode
        n_users = train_full.n_users
        print(f"训练数据集: {args.data}, 测试数据集: {args.test_data}")
        print(f"数据集标签模式: {label_mode}, 类别数: {num_classes}, 用户数: {n_users}")
        print(f"训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}, 测试样本数: {len(test_full)}")
    else:
        # 原有逻辑：从单个数据集中分割
        full = SchedulingNPZDataset(args.data, normalize=True, precompute=True)
        num_classes = full.num_classes
        label_mode = full.label_mode
        n_users = full.n_users
        print(f"数据集标签模式: {label_mode}, 类别数: {num_classes}, 用户数: {n_users}")
        
        train_ds, val_ds = split_dataset(full, val_ratio=args.val_ratio, seed=args.seed)
        test_loader = None  # 不使用独立的测试集

    dl_common = dict(num_workers=args.num_workers)
    if device == "cuda":
        dl_common["pin_memory"] = True
    if args.num_workers > 0:
        if args.prefetch_factor and args.prefetch_factor > 0:
            dl_common["prefetch_factor"] = args.prefetch_factor
        dl_common["persistent_workers"] = bool(args.persistent_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_common)

    model = build_model(args.model, input_dim=6, num_classes=num_classes, n_users=n_users).to(device)
    print(f"模型: {args.model}, 输入维度: 6, 输出类别数: {num_classes}, 用户数: {n_users}")
    
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

    # 初始化早停和训练历史记录
    early_stopping = None if args.no_early_stop else EarlyStopping(
        patience=args.patience, 
        min_delta=args.min_delta, 
        mode='max'
    )
    
    history = TrainingHistory(
        model_name=f"{args.model}_{label_mode}",
        save_dir=args.results_dir
    )

    best_acc = -1.0
    scaler = torch.amp.GradScaler('cuda',enabled=args.amp) if device == "cuda" else None
    non_blocking = (device == "cuda")
    epoch = 0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optim, scheduler, device,
            epoch=epoch,
            amp_enabled=bool(args.amp), scaler=scaler, non_blocking=non_blocking
        )
        va_loss, va_acc, va_top5, va_top10 = evaluate(
            model, val_loader, device, epoch=epoch,
            amp_enabled=bool(args.amp), non_blocking=non_blocking
        )
        current_lr = optim.param_groups[0]['lr']
        
        # 记录训练历史
        history.add_epoch(epoch, tr_loss, tr_acc, va_loss, va_acc, va_top5, va_top10, current_lr)
        
        tqdm.write(f"[Epoch {epoch:02d}] Train loss={tr_loss:.4f} acc={tr_acc:.4f} | Val loss={va_loss:.4f} acc={va_acc:.4f} top5={va_top5:.4f} top10={va_top10:.4f} | LR={current_lr:.6f}")
        
        # 保存最佳模型
        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(dict(
                model=args.model, state_dict=model.state_dict(),
                cfg=dict(n_users=n_users, num_classes=num_classes, label_mode=label_mode),
                best_epoch=epoch,
                best_val_acc=best_acc
            ), args.save)
            tqdm.write(f"  -> 保存最佳模型到: {args.save} (val_acc={best_acc:.4f})")
        
        # 早停检查
        if early_stopping is not None:
            if early_stopping(va_acc):
                tqdm.write(f"早停,验证准确率在 {early_stopping.patience} 个epoch内未提升")
                tqdm.write(f"最佳验证准确率: {early_stopping.best_value:.4f}")
                break

    # 保存训练历史
    history.save()
    
    print(f"\n训练完成")
    print(f"最佳验证准确率 Top-1={best_acc:.4f}")
    if early_stopping is not None and early_stopping.early_stop:
        print(f"提前停止于第 {epoch} 个epoch (早停耐心值: {args.patience})")
    
    # 如果有独立的测试集，进行最终测试
    if test_loader is not None:
        print("\n加载最佳模型进行测试...")
        checkpoint = torch.load(args.save, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        te_loss, te_acc, te_top5, te_top10 = evaluate(
            model, test_loader, device, epoch="Test",
            amp_enabled=bool(args.amp), non_blocking=non_blocking
        )
        print(f"测试结果: loss={te_loss:.4f} acc={te_acc:.4f} top5={te_top5:.4f} top10={te_top10:.4f}")
        
        # 将测试结果添加到历史记录
        history.add_test_results(te_loss, te_acc, te_top5, te_top10)
        history.save()

if __name__ == "__main__":
    main()
