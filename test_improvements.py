# -*- coding: utf-8 -*-
"""测试改进后的模型训练"""
import numpy as np
import torch
from dataset import SchedulingNPZDataset
from model import build_model

def test_data_normalization():
    """测试数据归一化效果"""
    print("=" * 60)
    print("测试1: 数据归一化效果")
    print("=" * 60)
    
    # 不归一化
    ds_raw = SchedulingNPZDataset("data/train.npz", normalize=False)
    sample_raw, label = ds_raw[0]
    
    # 归一化
    ds_norm = SchedulingNPZDataset("data/train.npz", normalize=True)
    sample_norm, _ = ds_norm[0]
    
    print(f"\n原始数据范围:")
    print(f"  最小值: {sample_raw.min().item():.2f}")
    print(f"  最大值: {sample_raw.max().item():.2f}")
    print(f"  均值: {sample_raw.mean().item():.2f}")
    print(f"  标准差: {sample_raw.std().item():.2f}")
    
    print(f"\n归一化后范围:")
    print(f"  最小值: {sample_norm.min().item():.2f}")
    print(f"  最大值: {sample_norm.max().item():.2f}")
    print(f"  均值: {sample_norm.mean().item():.2f}")
    print(f"  标准差: {sample_norm.std().item():.2f}")
    
    print(f"\n✓ 归一化成功！数据现在在相近的尺度上")

def test_model_forward():
    """测试模型前向传播是否产生 NaN"""
    print("\n" + "=" * 60)
    print("测试2: 模型前向传播稳定性")
    print("=" * 60)
    
    ds = SchedulingNPZDataset("data/train.npz", normalize=True)
    batch_x = torch.stack([ds[i][0] for i in range(32)])  # [32, 128, 6]
    
    models = ["bilstm", "mlp", "transformer"]
    
    for model_name in models:
        print(f"\n测试模型: {model_name}")
        model = build_model(model_name, input_dim=6)
        model.eval()
        
        with torch.no_grad():
            logits = model(batch_x)
        
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        
        print(f"  输出形状: {logits.shape}")
        print(f"  输出范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"  是否有 NaN: {'❌ 是' if has_nan else '✓ 否'}")
        print(f"  是否有 Inf: {'❌ 是' if has_inf else '✓ 否'}")
        
        if not has_nan and not has_inf:
            print(f"  ✓ {model_name} 前向传播稳定")

def test_gradient_flow():
    """测试梯度流动"""
    print("\n" + "=" * 60)
    print("测试3: 梯度流动")
    print("=" * 60)
    
    ds = SchedulingNPZDataset("data/train.npz", normalize=True)
    batch_x = torch.stack([ds[i][0] for i in range(32)])
    batch_y = torch.tensor([ds[i][1] for i in range(32)])
    
    models = ["bilstm", "mlp", "transformer"]
    
    for model_name in models:
        print(f"\n测试模型: {model_name}")
        model = build_model(model_name, input_dim=6)
        model.train()
        
        # 前向传播
        logits = model(batch_x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, batch_y)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if torch.isnan(param.grad).any():
                    print(f"  ❌ {name} 的梯度包含 NaN")
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  是否有 NaN Loss: {'❌ 是' if np.isnan(loss.item()) else '✓ 否'}")
        print(f"  平均梯度范数: {np.mean(grad_norms):.6f}")
        print(f"  最大梯度范数: {np.max(grad_norms):.6f}")
        
        if not np.isnan(loss.item()) and np.max(grad_norms) < 100:
            print(f"  ✓ {model_name} 梯度流动正常")

if __name__ == "__main__":
    print("\n" + "🔬 开始测试改进方案" + "\n")
    
    test_data_normalization()
    test_model_forward()
    test_gradient_flow()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("=" * 60)
    print("\n建议:")
    print("1. 现在可以运行训练命令:")
    print("   python train.py --model transformer --epochs 50")
    print("2. 期望 Transformer 准确率提升到 30%+")
    print("3. Loss 应该不再出现 NaN")
