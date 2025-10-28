# -*- coding: utf-8 -*-
"""
可视化训练历史：支持对比多个模型run的曲线。
用法示例：
  python visualize.py --results results
  python visualize.py --files results/bilstm_user_history.json results/mlp_user_history.json
生成图片会保存在 results/plots/ 下。
"""
import argparse
import json
import os
from glob import glob
from typing import List

import matplotlib
matplotlib.use('Agg')  # 服务器/无显示环境安全
import matplotlib.pyplot as plt
import seaborn as sns


def load_histories(files: List[str]):
    runs = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as rf:
                hist = json.load(rf)
                hist['__file'] = f
                runs.append(hist)
        except Exception as e:
            print(f"跳过 {f}: {e}")
    return runs


def plot_metric(runs, metric: str, ylabel: str, out_path: str):
    plt.figure(figsize=(8, 5))
    sns.set_style('whitegrid')
    for h in runs:
        name = os.path.splitext(os.path.basename(h.get('__file__', h.get('model', 'run'))))[0]
        x = h.get('epochs', list(range(1, len(h.get(metric, [])) + 1)))
        y = h.get(metric, [])
        if not y:
            continue
        plt.plot(x, y, label=name)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"保存图像: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=str, default='results', help='训练历史目录')
    ap.add_argument('--files', type=str, nargs='*', default=None, help='指定若干JSON文件进行对比')
    args = ap.parse_args()

    if args.files is None or len(args.files) == 0:
        pattern = os.path.join(args.results, '*_history.json')
        files = sorted(glob(pattern))
        if not files:
            print(f"未在 {args.results} 找到 *_history.json 文件。")
            return
    else:
        files = args.files

    runs = load_histories(files)
    if not runs:
        print("没有可用的训练历史")
        return

    plot_dir = os.path.join(args.results, 'plots')
    # 准确率类
    plot_metric(runs, 'train_acc', 'Train Accuracy', os.path.join(plot_dir, 'train_acc.png'))
    plot_metric(runs, 'val_acc', 'Val Accuracy (Top-1)', os.path.join(plot_dir, 'val_acc.png'))
    plot_metric(runs, 'val_top5', 'Val Top-5', os.path.join(plot_dir, 'val_top5.png'))
    plot_metric(runs, 'val_top10', 'Val Top-10', os.path.join(plot_dir, 'val_top10.png'))
    # 损失类
    plot_metric(runs, 'train_loss', 'Train Loss', os.path.join(plot_dir, 'train_loss.png'))
    plot_metric(runs, 'val_loss', 'Val Loss', os.path.join(plot_dir, 'val_loss.png'))
    # 学习率
    plot_metric(runs, 'learning_rate', 'Learning Rate', os.path.join(plot_dir, 'lr.png'))


if __name__ == '__main__':
    main()
