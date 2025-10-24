
# -*- coding: utf-8 -*-
import argparse, os, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config import CFG
from dataset import SchedulingNPZDataset, split_dataset
from model import build_model

def set_seed(seed: int = 2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, optim, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    losses = []
    all_pred, all_tgt = [], []
    for feats, label in loader:
        feats = feats.to(device)          # [B, 128, 6]
        label = label.to(device)          # [B]
        optim.zero_grad()
        logits = model(feats)             # [B, 128]
        loss = crit(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)
        all_pred.append(pred.detach().cpu().numpy())
        all_tgt.append(label.detach().cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_tgt = np.concatenate(all_tgt)
    acc = accuracy_score(all_tgt, all_pred)
    return float(np.mean(losses)), float(acc)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    losses = []
    all_pred, all_tgt = [], []
    for feats, label in loader:
        feats = feats.to(device)
        label = label.to(device)
        logits = model(feats)
        loss = crit(logits, label)
        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)
        all_pred.append(pred.detach().cpu().numpy())
        all_tgt.append(label.detach().cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_tgt = np.concatenate(all_tgt)
    acc = accuracy_score(all_tgt, all_pred)
    return float(np.mean(losses)), float(acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/train.npz", help="npz 数据集路径")
    ap.add_argument("--model", type=str, default="bilstm", choices=["bilstm", "mlp", "transformer"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--save", type=str, default="checkpoints/best.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据
    full = SchedulingNPZDataset(args.data)
    train_ds, val_ds = split_dataset(full, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 模型
    model = build_model(args.model, input_dim=6).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:02d}] Train loss={tr_loss:.4f} acc={tr_acc:.4f} | Val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(dict(
                model=args.model, state_dict=model.state_dict(),
                cfg=dict(N_USERS=CFG.N_USERS)
            ), args.save)
            print(f"  -> 保存最佳模型到: {args.save} (val_acc={best_acc:.4f})")

    print(f"训练完成。最佳验证准确率: {best_acc:.4f}")

if __name__ == "__main__":
    main()
