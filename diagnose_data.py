# -*- coding: utf-8 -*-
"""诊断数据集，检查数据分布和潜在问题"""
import numpy as np
import argparse
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/train.npz")
    args = ap.parse_args()
    
    print(f"Loading {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    X = data["X"]  # [N, 128, 3, 2]
    y = data["y"]  # [N]
    
    print("\n=== 数据集基本信息 ===")
    print(f"样本数: {X.shape[0]}")
    print(f"用户数: {X.shape[1]}")
    print(f"业务数: {X.shape[2]}")
    print(f"特征数 (per user per service): {X.shape[3]}")
    
    print("\n=== 标签分布 ===")
    label_counts = Counter(y)
    print(f"唯一标签数: {len(label_counts)}")
    print(f"标签范围: {min(y)} ~ {max(y)}")
    top_10 = label_counts.most_common(10)
    print("Top 10 最常见的用户ID:")
    for uid, count in top_10:
        print(f"  User {uid}: {count} ({count/len(y)*100:.2f}%)")
    
    # 检查是否存在类别不平衡问题
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    print(f"\n类别平衡性:")
    print(f"  最多: {max_count} 样本")
    print(f"  最少: {min_count} 样本")
    print(f"  比例: {max_count/min_count:.2f}:1")
    
    print("\n=== 特征统计 ===")
    # Buffer bytes (index 0)
    buffers = X[:, :, :, 0]  # [N, 128, 3]
    print("Buffer bytes (VO/VI/BE):")
    for s, name in enumerate(['VO', 'VI', 'BE']):
        buf_s = buffers[:, :, s]
        non_zero = buf_s[buf_s > 0]
        print(f"  {name}:")
        print(f"    非零比例: {len(non_zero) / buf_s.size * 100:.2f}%")
        if len(non_zero) > 0:
            print(f"    范围: {non_zero.min():.0f} ~ {non_zero.max():.0f}")
            print(f"    均值: {non_zero.mean():.0f}")
            print(f"    标准差: {non_zero.std():.0f}")
    
    # Wait ms (index 1)
    waits = X[:, :, :, 1]  # [N, 128, 3]
    print("\nWait ms (VO/VI/BE):")
    for s, name in enumerate(['VO', 'VI', 'BE']):
        wait_s = waits[:, :, s]
        print(f"  {name}:")
        print(f"    范围: {wait_s.min():.2f} ~ {wait_s.max():.2f}")
        print(f"    均值: {wait_s.mean():.2f}")
        print(f"    标准差: {wait_s.std():.2f}")
    
    print("\n=== 检查数据质量 ===")
    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()
    print(f"包含 NaN: {has_nan}")
    print(f"包含 Inf: {has_inf}")
    
    # 检查每个样本是否至少有一个用户有缓存
    has_buffer = (buffers > 0).any(axis=(1, 2))  # [N]
    print(f"所有样本都有至少一个缓存: {has_buffer.all()}")
    if not has_buffer.all():
        print(f"  警告: {(~has_buffer).sum()} 个样本没有任何缓存!")
    
    # 检查标签的有效性 (应该对应有缓存的用户)
    print("\n=== 检查标签有效性 ===")
    invalid_labels = 0
    for i in range(min(1000, len(y))):  # 检查前1000个样本
        uid = y[i]
        user_has_buf = (X[i, uid, :, 0] > 0).any()
        if not user_has_buf:
            invalid_labels += 1
    
    if invalid_labels > 0:
        print(f"警告: 前1000个样本中有 {invalid_labels} 个标签指向没有缓存的用户!")
    else:
        print("前1000个样本的标签都有效 ✓")
    
    if 'meta' in data:
        print("\n=== 元数据 ===")
        meta = data['meta'].item()
        for k, v in meta.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
