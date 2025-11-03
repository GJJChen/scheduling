# -*- coding: utf-8 -*-
"""
测试脚本：加载训练好的模型，在指定测试集上进行评估
"""
import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import SchedulingNPZDataset
from model import build_model


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=False, non_blocking=False):
    """评估模型在测试集上的性能"""
    model.eval()
    crit = nn.CrossEntropyLoss()
    losses = []
    all_pred, all_tgt = [], []
    all_logits = []
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Testing")
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
    ap = argparse.ArgumentParser(description="测试训练好的调度模型")
    ap.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径 (.pt文件)")
    ap.add_argument("--test-data", type=str, required=True, help="测试数据集路径 (.npz文件)")
    ap.add_argument("--batch-size", type=int, default=256, help="批次大小")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    ap.add_argument("--amp", action="store_true", help="使用混合精度推理")
    ap.add_argument("--output", type=str, help="结果输出JSON文件路径（可选）")
    args = ap.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型检查点不存在: {args.checkpoint}")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"测试数据集不存在: {args.test_data}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型检查点
    print(f"加载模型检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 提取模型配置
    model_type = checkpoint['model']
    cfg = checkpoint['cfg']
    n_users = cfg['n_users']
    num_classes = cfg['num_classes']
    label_mode = cfg['label_mode']
    
    print(f"模型类型: {model_type}")
    print(f"标签模式: {label_mode}, 类别数: {num_classes}, 用户数: {n_users}")
    
    if 'train_scenario' in checkpoint:
        print(f"训练场景: {checkpoint['train_scenario']}")
    if 'test_scenario' in checkpoint:
        print(f"原测试场景: {checkpoint['test_scenario']}")
    if 'best_epoch' in checkpoint:
        print(f"最佳epoch: {checkpoint['best_epoch']}")
    if 'best_val_acc' in checkpoint:
        print(f"最佳验证准确率: {checkpoint['best_val_acc']:.4f}")
    
    # 加载测试数据集
    print(f"\n加载测试数据集: {args.test_data}")
    
    # 如果模型保存了归一化参数，使用它们
    norm_params = checkpoint.get('norm_params', None)
    if norm_params is not None:
        print("使用模型保存的归一化参数")
    
    test_dataset = SchedulingNPZDataset(
        args.test_data,
        normalize=True,
        precompute=True,
        norm_params=norm_params
    )
    
    # 检查数据集与模型的兼容性
    if test_dataset.label_mode != label_mode:
        raise ValueError(f"测试集标签模式 ({test_dataset.label_mode}) 与模型不匹配 ({label_mode})")
    if test_dataset.num_classes != num_classes:
        raise ValueError(f"测试集类别数 ({test_dataset.num_classes}) 与模型不匹配 ({num_classes})")
    if test_dataset.n_users != n_users:
        raise ValueError(f"测试集用户数 ({test_dataset.n_users}) 与模型不匹配 ({n_users})")
    
    print(f"测试样本数: {len(test_dataset)}")
    
    # 构建DataLoader
    dl_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers
    }
    if device == "cuda":
        dl_kwargs['pin_memory'] = True
    
    test_loader = DataLoader(test_dataset, **dl_kwargs)
    
    # 构建模型
    print(f"\n构建模型...")
    model = build_model(
        model_type,
        input_dim=6,
        num_classes=num_classes,
        n_users=n_users
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])
    print("模型权重加载完成")
    
    # 评估模型
    print(f"\n开始评估...")
    test_loss, test_acc, test_top5, test_top10 = evaluate(
        model,
        test_loader,
        device,
        amp_enabled=args.amp,
        non_blocking=(device == "cuda")
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果:")
    print(f"  损失 (Loss):        {test_loss:.4f}")
    print(f"  Top-1 准确率:        {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Top-5 准确率:        {test_top5:.4f} ({test_top5*100:.2f}%)")
    print(f"  Top-10 准确率:       {test_top10:.4f} ({test_top10*100:.2f}%)")
    print("="*60)
    
    # 保存结果到JSON（如果指定了输出路径）
    if args.output:
        results = {
            'checkpoint': args.checkpoint,
            'test_data': args.test_data,
            'model_type': model_type,
            'label_mode': label_mode,
            'num_classes': num_classes,
            'n_users': n_users,
            'test_samples': len(test_dataset),
            'results': {
                'loss': test_loss,
                'top1_acc': test_acc,
                'top5_acc': test_top5,
                'top10_acc': test_top10
            }
        }
        
        if 'train_scenario' in checkpoint:
            results['train_scenario'] = checkpoint['train_scenario']
        if 'best_epoch' in checkpoint:
            results['best_epoch'] = checkpoint['best_epoch']
        if 'best_val_acc' in checkpoint:
            results['best_val_acc'] = checkpoint['best_val_acc']
        
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
