# -*- coding: utf-8 -*-
"""
将CSV格式的数据集转换为npz格式

输入CSV格式：
- input.csv: 第一行为序号，其余每行是 [128, 3, 2] 张量展平后的768个值
- output.csv: 第一行为序号，其余每行包含两列：用户ID(0-127), 业务ID(0-2)

输出格式：
- npz文件，包含：
  - X: shape [N, 128, 3, 2] 的输入特征
  - y: shape [N] 的标签（user * 3 + service 或单独编码）
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import CFG


def convert_csv_to_npz(input_csv, output_csv, out_npz, label_mode='combined'):
    """
    转换CSV数据集为npz格式
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        out_npz: 输出npz文件路径
        label_mode: 标签编码模式
            - 'combined': y = user * 3 + service (0-383)
            - 'user': y = user (0-127)
    """
    print(f"读取输入文件: {input_csv}")
    # 读取输入CSV（跳过第一行序号列）
    df_input = pd.read_csv(input_csv)
    
    # 假设第一列是序号，从第二列开始是数据
    if df_input.shape[1] == 769:  # 序号 + 768个数据
        print("检测到第一列为序号，将跳过")
        input_data = df_input.iloc[:, 1:].values  # 跳过第一列
    elif df_input.shape[1] == 768:  # 只有数据
        print("未检测到序号列")
        input_data = df_input.values
    else:
        raise ValueError(f"输入CSV列数不符合预期：期望768或769列，实际{df_input.shape[1]}列")
    
    print(f"读取输出文件: {output_csv}")
    # 读取输出CSV
    df_output = pd.read_csv(output_csv)
    
    # 假设第一列是序号，后面是用户ID和业务ID
    if df_output.shape[1] == 3:  # 序号 + 用户 + 业务
        print("检测到第一列为序号，将跳过")
        output_data = df_output.iloc[:, 1:].values  # 跳过第一列
    elif df_output.shape[1] == 2:  # 只有用户和业务
        print("未检测到序号列")
        output_data = df_output.values
    else:
        raise ValueError(f"输出CSV列数不符合预期：期望2或3列，实际{df_output.shape[1]}列")
    
    # 检查行数是否匹配
    n_samples = input_data.shape[0]
    if output_data.shape[0] != n_samples:
        raise ValueError(f"输入输出行数不匹配：输入{n_samples}行，输出{output_data.shape[0]}行")
    
    print(f"总样本数: {n_samples}")
    
    # 转换输入数据：从展平的768维向量恢复为 [128, 3, 2] 张量
    print("重塑输入数据为 [N, 128, 3, 2] 格式...")
    X = np.zeros((n_samples, CFG.N_USERS, 3, 2), dtype=np.float32)
    
    for i in tqdm(range(n_samples), desc="处理输入数据", ncols=100):
        # 将 768 维向量重塑为 [128, 3, 2]
        X[i] = input_data[i].reshape(CFG.N_USERS, 3, 2)
    
    # 转换输出数据：提取用户ID和业务ID
    print("处理输出标签...")
    users = output_data[:, 0].astype(np.int64)  # 用户ID (0-127)
    services = output_data[:, 1].astype(np.int64)  # 业务ID (0-2)
    
    # 验证范围
    if np.any((users < 0) | (users >= CFG.N_USERS)):
        raise ValueError(f"用户ID超出范围 [0, {CFG.N_USERS-1}]")
    if np.any((services < 0) | (services >= 3)):
        raise ValueError("业务ID超出范围 [0, 2]")
    
    # 根据模式生成标签
    if label_mode == 'combined':
        # 组合编码：y = user * 3 + service (0-383)
        y = users * 3 + services
        print(f"使用组合标签模式: y = user * 3 + service (范围: 0-{CFG.N_USERS*3-1})")
    elif label_mode == 'user':
        # 只使用用户ID
        y = users
        print(f"使用用户标签模式: y = user (范围: 0-{CFG.N_USERS-1})")
    else:
        raise ValueError(f"未知的标签模式: {label_mode}")
    
    # 统计信息
    print("\n数据统计:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y range: [{y.min()}, {y.max()}]")
    print(f"  y 唯一值数量: {len(np.unique(y))}")
    
    # 业务分布统计
    print("\n业务分布:")
    for s in range(3):
        count = np.sum(services == s)
        print(f"  {CFG.SERVICES[s]}: {count} ({count/n_samples*100:.2f}%)")
    
    # 保存为npz格式
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    print(f"\n保存到: {out_npz}")
    
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        meta=dict(
            services=CFG.SERVICES,
            attrs=CFG.ATTRS,
            n_users=CFG.N_USERS,
            label_mode=label_mode,
            n_samples=n_samples
        )
    )
    
    print("转换完成！")
    
    # 验证保存的数据
    print("\n验证保存的数据...")
    loaded = np.load(out_npz, allow_pickle=True)
    print(f"  X shape: {loaded['X'].shape}")
    print(f"  y shape: {loaded['y'].shape}")
    print(f"  Meta: {loaded['meta'].item()}")


def main():
    parser = argparse.ArgumentParser(description="将CSV格式数据集转换为npz格式")
    parser.add_argument("--input", type=str, required=True, 
                        help="输入CSV文件路径（展平的768维特征）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出CSV文件路径（用户ID和业务ID）")
    parser.add_argument("--out-npz", type=str, default="data/train.npz",
                        help="输出npz文件路径（默认: data/train.npz）")
    parser.add_argument("--label-mode", type=str, default="combined",
                        choices=['combined', 'user'],
                        help="标签编码模式：combined(user*3+service) 或 user(仅用户ID)")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    if not os.path.exists(args.output):
        raise FileNotFoundError(f"输出文件不存在: {args.output}")
    
    # 执行转换
    convert_csv_to_npz(
        input_csv=args.input,
        output_csv=args.output,
        out_npz=args.out_npz,
        label_mode=args.label_mode
    )


if __name__ == "__main__":
    main()
