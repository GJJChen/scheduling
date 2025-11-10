# -*- coding: utf-8 -*-
"""
合并多个CSV文件为单个NPZ文件的示例脚本
"""

import subprocess
import os

# 示例1: 合并多个CSV文件对（使用逗号分隔）
def example_merge_multiple_files():
    """
    合并多个input/output CSV文件对为单个npz文件
    """
    command = [
        "python", "convert_csv_to_npz.py",
        "--merge",
        "--input", "data/input1.csv,data/input2.csv,data/input3.csv",
        "--output", "data/output1.csv,data/output2.csv,data/output3.csv",
        "--out-npz", "data/merged_train.npz",
        "--label-mode", "combined"
    ]
    
    print("示例1: 合并多个CSV文件对")
    print("命令:", " ".join(command))
    # subprocess.run(command)


# 示例2: 合并两个文件（简单示例）
def example_merge_two_files():
    """
    合并两个CSV文件对
    """
    command = [
        "python", "convert_csv_to_npz.py",
        "--merge",
        "--input", "data/train_input.csv,data/val_input.csv",
        "--output", "data/train_output.csv,data/val_output.csv",
        "--out-npz", "data/combined.npz",
        "--label-mode", "combined",
        "--swap-last-dim"
    ]
    
    print("\n示例2: 合并训练集和验证集")
    print("命令:", " ".join(command))
    # subprocess.run(command)


# 示例3: 单文件转换（原有功能保持不变）
def example_single_file():
    """
    单个CSV文件对转换（不合并）
    """
    command = [
        "python", "convert_csv_to_npz.py",
        "--input", "data/input.csv",
        "--output", "data/output.csv",
        "--out-npz", "data/train.npz",
        "--label-mode", "combined"
    ]
    
    print("\n示例3: 单文件转换")
    print("命令:", " ".join(command))
    # subprocess.run(command)


if __name__ == "__main__":
    print("=" * 60)
    print("CSV合并为NPZ文件的使用示例")
    print("=" * 60)
    
    example_merge_multiple_files()
    example_merge_two_files()
    example_single_file()
    
    print("\n" + "=" * 60)
    print("使用说明:")
    print("1. 使用 --merge 参数启用合并模式")
    print("2. 使用逗号分隔多个输入和输出文件路径")
    print("3. 所有文件对的数据会合并到单个npz文件中")
    print("4. 输入和输出文件数量必须匹配")
    print("=" * 60)
