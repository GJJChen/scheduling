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


def convert_csv_to_npz(input_csv, output_csv, out_npz, label_mode='combined', swap_last_dim=False):
    """
    转换CSV数据集为npz格式
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        out_npz: 输出npz文件路径
        label_mode: 标签编码模式
            - 'combined': y = user * 3 + service (0-383)
            - 'user': y = user (0-127)
        swap_last_dim: 是否交换最后一个维度（大小为2）的两个属性的顺序
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
    
    # 如果需要，交换最后一个维度的两个属性的顺序
    if swap_last_dim:
        print("交换最后一个维度的两个属性顺序 (.., 2) -> (.., [1, 0])")
        X = X[..., [1, 0]]

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
    
    # 根据是否交换调整元数据中的属性顺序
    attrs_meta = list(getattr(CFG, 'ATTRS', []))
    if swap_last_dim and len(attrs_meta) == 2:
        attrs_meta = attrs_meta[::-1]

    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        meta=dict(
            services=CFG.SERVICES,
            attrs=attrs_meta,
            n_users=CFG.N_USERS,
            label_mode=label_mode,
            n_samples=n_samples,
            swapped_last_dim=bool(swap_last_dim)
        )
    )
    
    print("转换完成！")
    
    # 验证保存的数据
    print("\n验证保存的数据...")
    loaded = np.load(out_npz, allow_pickle=True)
    print(f"  X shape: {loaded['X'].shape}")
    print(f"  y shape: {loaded['y'].shape}")
    print(f"  Meta: {loaded['meta'].item()}")


def merge_csv_to_npz(csv_pairs, out_npz, label_mode='combined', swap_last_dim=False):
    """
    合并多个CSV文件对为单个npz格式
    
    Args:
        csv_pairs: CSV文件对列表，每个元素为 (input_csv, output_csv)
        out_npz: 输出npz文件路径
        label_mode: 标签编码模式
        swap_last_dim: 是否交换最后一个维度的顺序
    """
    all_X = []
    all_y = []
    total_samples = 0
    
    print(f"开始合并 {len(csv_pairs)} 个CSV文件对...")
    
    for idx, (input_csv, output_csv) in enumerate(csv_pairs, 1):
        print(f"\n[{idx}/{len(csv_pairs)}] 处理文件对:")
        print(f"  输入: {input_csv}")
        print(f"  输出: {output_csv}")
        
        # 读取输入CSV
        df_input = pd.read_csv(input_csv)
        
        # 处理序号列
        if df_input.shape[1] == 769:
            input_data = df_input.iloc[:, 1:].values
        elif df_input.shape[1] == 768:
            input_data = df_input.values
        else:
            raise ValueError(f"输入CSV列数不符合预期：期望768或769列，实际{df_input.shape[1]}列")
        
        # 读取输出CSV
        df_output = pd.read_csv(output_csv)
        
        if df_output.shape[1] == 3:
            output_data = df_output.iloc[:, 1:].values
        elif df_output.shape[1] == 2:
            output_data = df_output.values
        else:
            raise ValueError(f"输出CSV列数不符合预期：期望2或3列，实际{df_output.shape[1]}列")
        
        # 检查行数匹配
        n_samples = input_data.shape[0]
        if output_data.shape[0] != n_samples:
            raise ValueError(f"输入输出行数不匹配：输入{n_samples}行，输出{output_data.shape[0]}行")
        
        print(f"  样本数: {n_samples}")
        
        # 转换输入数据
        X = np.zeros((n_samples, CFG.N_USERS, 3, 2), dtype=np.float32)
        for i in range(n_samples):
            X[i] = input_data[i].reshape(CFG.N_USERS, 3, 2)
        
        # 交换维度（如果需要）
        if swap_last_dim:
            X = X[..., [1, 0]]
        
        # 转换输出数据
        users = output_data[:, 0].astype(np.int64)
        services = output_data[:, 1].astype(np.int64)
        
        # 验证范围
        if np.any((users < 0) | (users >= CFG.N_USERS)):
            raise ValueError(f"用户ID超出范围 [0, {CFG.N_USERS-1}]")
        if np.any((services < 0) | (services >= 3)):
            raise ValueError("业务ID超出范围 [0, 2]")
        
        # 生成标签
        if label_mode == 'combined':
            y = users * 3 + services
        elif label_mode == 'user':
            y = users
        else:
            raise ValueError(f"未知的标签模式: {label_mode}")
        
        all_X.append(X)
        all_y.append(y)
        total_samples += n_samples
    
    # 合并所有数据
    print(f"\n合并所有数据...")
    X_merged = np.concatenate(all_X, axis=0)
    y_merged = np.concatenate(all_y, axis=0)
    
    # 统计信息
    print("\n合并后的数据统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  X shape: {X_merged.shape}")
    print(f"  y shape: {y_merged.shape}")
    print(f"  X range: [{X_merged.min():.2f}, {X_merged.max():.2f}]")
    print(f"  y range: [{y_merged.min()}, {y_merged.max()}]")
    print(f"  y 唯一值数量: {len(np.unique(y_merged))}")
    
    # 业务分布统计
    print("\n业务分布:")
    if label_mode == 'combined':
        services_merged = y_merged % 3
    else:
        # 需要从原始数据重新获取服务信息
        services_merged = np.concatenate([np.full(len(y), -1) for y in all_y])
        offset = 0
        for input_csv, output_csv in csv_pairs:
            df_output = pd.read_csv(output_csv)
            if df_output.shape[1] == 3:
                output_data = df_output.iloc[:, 1:].values
            else:
                output_data = df_output.values
            services = output_data[:, 1].astype(np.int64)
            services_merged[offset:offset+len(services)] = services
            offset += len(services)
    
    for s in range(3):
        count = np.sum(services_merged == s)
        print(f"  {CFG.SERVICES[s]}: {count} ({count/total_samples*100:.2f}%)")
    
    # 保存为npz格式
    os.makedirs(os.path.dirname(out_npz) if os.path.dirname(out_npz) else '.', exist_ok=True)
    print(f"\n保存到: {out_npz}")
    
    # 准备元数据
    attrs_meta = list(getattr(CFG, 'ATTRS', []))
    if swap_last_dim and len(attrs_meta) == 2:
        attrs_meta = attrs_meta[::-1]
    
    np.savez_compressed(
        out_npz,
        X=X_merged,
        y=y_merged,
        meta=dict(
            services=CFG.SERVICES,
            attrs=attrs_meta,
            n_users=CFG.N_USERS,
            label_mode=label_mode,
            n_samples=total_samples,
            swapped_last_dim=bool(swap_last_dim),
            source_files=len(csv_pairs)
        )
    )
    
    print("合并完成！")
    
    # 验证保存的数据
    print("\n验证保存的数据...")
    loaded = np.load(out_npz, allow_pickle=True)
    print(f"  X shape: {loaded['X'].shape}")
    print(f"  y shape: {loaded['y'].shape}")
    print(f"  Meta: {loaded['meta'].item()}")


def main():
    parser = argparse.ArgumentParser(description="将CSV格式数据集转换为npz格式")
    parser.add_argument("--input", type=str,
                        help="输入CSV文件路径（展平的768维特征），多个文件用逗号分隔")
    parser.add_argument("--output", type=str,
                        help="输出CSV文件路径（用户ID和业务ID），多个文件用逗号分隔")
    parser.add_argument("--data-dir", type=str,
                        help="包含多个CSV数据集的文件夹路径")
    parser.add_argument("--out-npz", type=str, default="data/train.npz",
                        help="输出npz文件路径（单个文件模式）")
    parser.add_argument("--label-mode", type=str, default="combined",
                        choices=['combined', 'user'],
                        help="标签编码模式：combined(user*3+service) 或 user(仅用户ID)")
    parser.add_argument("--swap-last-dim", action="store_true",
                        help="是否交换最后一个维度（大小为2）的两个属性的顺序")
    parser.add_argument("--merge", action="store_true",
                        help="合并多个CSV文件对到单个npz文件")

    parser.add_argument("--merge", action="store_true",
                        help="合并多个CSV文件对到单个npz文件")

    args = parser.parse_args()
    
    # 检查参数
    if args.merge:
        # 合并模式：支持通过 --input 和 --output 指定多个文件
        if not args.input or not args.output:
            raise ValueError("合并模式需要指定 --input 和 --output 参数")
        
        # 分隔文件路径（支持逗号分隔）
        input_files = [f.strip() for f in args.input.split(',')]
        output_files = [f.strip() for f in args.output.split(',')]
        
        if len(input_files) != len(output_files):
            raise ValueError(f"输入输出文件数量不匹配：{len(input_files)} vs {len(output_files)}")
        
        # 检查文件是否存在
        for f in input_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"输入文件不存在: {f}")
        for f in output_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"输出文件不存在: {f}")
        
        # 创建文件对列表
        csv_pairs = list(zip(input_files, output_files))
        
        # 执行合并
        merge_csv_to_npz(
            csv_pairs=csv_pairs,
            out_npz=args.out_npz,
            label_mode=args.label_mode,
            swap_last_dim=args.swap_last_dim
        )
        
    elif args.data_dir:
        # 多文件模式
        if not os.path.isdir(args.data_dir):
            raise ValueError(f"数据文件夹不存在: {args.data_dir}")
        
        # 扫描文件夹中的CSV文件
        import glob
        import re
        
        input_files = glob.glob(os.path.join(args.data_dir, "input_data_*_set.csv"))
        output_files = glob.glob(os.path.join(args.data_dir, "output_data_*_set.csv"))
        
        # 匹配输入输出文件对
        file_pairs = {}
        pattern = re.compile(r'(input|output)_data_([^_]+)_(.+)_set\.csv')
        
        for f in input_files + output_files:
            basename = os.path.basename(f)
            match = pattern.match(basename)
            if match:
                io_type, business, scenario = match.groups()
                key = f"{business}_{scenario}"
                if key not in file_pairs:
                    file_pairs[key] = {}
                file_pairs[key][io_type] = f
        
        # 处理每个数据集
        for key, files in file_pairs.items():
            if 'input' in files and 'output' in files:
                business, scenario = key.split('_', 1)
                out_npz = os.path.join("data", f"{business}_{scenario}.npz")
                print(f"\n处理数据集: {key}")
                convert_csv_to_npz(
                    input_csv=files['input'],
                    output_csv=files['output'],
                    out_npz=out_npz,
                    label_mode=args.label_mode,
                    swap_last_dim=args.swap_last_dim
                )
            else:
                print(f"警告: 数据集 {key} 缺少输入或输出文件")
        
    else:
        # 单文件模式（原有逻辑）
        if not args.input or not args.output:
            raise ValueError("单文件模式需要指定 --input 和 --output 参数")
        
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
            label_mode=args.label_mode,
            swap_last_dim=args.swap_last_dim
        )


if __name__ == "__main__":
    main()
