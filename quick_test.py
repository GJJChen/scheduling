# -*- coding: utf-8 -*-
"""快速测试脚本：检查模型是否能正常训练（无NaN）"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset import SchedulingNPZDataset, split_dataset
from model import build_model
from config import CFG

def quick_test():
    print("=== 快速测试脚本 ===\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    # 加载数据
    print("1. 加载数据...")
    try:
        full = SchedulingNPZDataset("data/train.npz", normalize=True)
        print(f"   数据集大小: {len(full)} 样本")
        
        # 测试一个样本
        feats, label = full[0]
        print(f"   特征形状: {feats.shape}, 标签: {label}")
        print(f"   特征范围: [{feats.min():.4f}, {feats.max():.4f}]")
        print(f"   特征均值: {feats.mean():.4f}, 标准差: {feats.std():.4f}")
        
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            print("   ❌ 数据中包含 NaN 或 Inf!")
            return False
        print("   ✓ 数据正常\n")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return False
    
    # 创建小批量数据
    train_ds, val_ds = split_dataset(full, val_ratio=0.1, seed=2025)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # 测试三种模型
    models = ["bilstm", "mlp", "transformer"]
    for model_name in models:
        print(f"2. 测试 {model_name.upper()} 模型...")
        try:
            model = build_model(model_name, input_dim=6).to(device)
            print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 测试前向传播
            model.eval()
            with torch.no_grad():
                feats_batch = torch.randn(4, 128, 6).to(device) * 0.5 + 0.5  # 模拟归一化后的数据
                logits = model(feats_batch)
                print(f"   前向传播输出形状: {logits.shape}")
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"   ❌ 输出包含 NaN 或 Inf!")
                    continue
                print(f"   输出范围: [{logits.min():.4f}, {logits.max():.4f}]")
            
            # 测试训练
            model.train()
            optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            
            print("   测试训练迭代...")
            batch_count = 0
            for feats, label in train_loader:
                feats = feats.to(device)
                label = label.to(device)
                
                optim.zero_grad()
                logits = model(feats)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"   ❌ Batch {batch_count}: 输出包含 NaN 或 Inf!")
                    break
                
                loss = crit(logits, label)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"   ❌ Batch {batch_count}: Loss 为 NaN 或 Inf!")
                    break
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                
                pred = torch.argmax(logits, dim=1)
                acc = (pred == label).float().mean().item()
                
                if batch_count == 0:
                    print(f"   Batch 0: loss={loss.item():.4f}, acc={acc:.4f}")
                
                batch_count += 1
                if batch_count >= 10:  # 测试10个batch
                    break
            
            if batch_count >= 10:
                print(f"   ✓ {model_name.upper()} 模型训练正常\n")
            
        except Exception as e:
            print(f"   ❌ {model_name.upper()} 测试失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=== 测试完成 ===")
    return True

if __name__ == "__main__":
    quick_test()
