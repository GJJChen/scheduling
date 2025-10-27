
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from config import CFG
from rules import schedule_one, VO, VI, BE, BUF, WAIT

def sample_snapshot(rng: np.random.Generator) -> np.ndarray:
    """随机生成一个 1ms 输入快照: shape=(N_users, 3, 2)。确保至少一个可调度候选。"""
    N = CFG.N_USERS
    snap = np.zeros((N, 3, 2), dtype=np.float32)

    # 逐用户逐业务采样 buffer & wait
    for u in range(N):
        for s, name in enumerate(CFG.SERVICES):
            # 是否有缓存
            if rng.random() < CFG.P_HAS_BUF[name]:
                lo, hi = CFG.BUF_RANGE[name]
                snap[u, s, BUF] = rng.integers(lo, hi+1)
            else:
                snap[u, s, BUF] = 0

            lo_w, hi_w = CFG.WAIT_RANGE[name]
            snap[u, s, WAIT] = rng.uniform(lo_w, hi_w)

    return snap

def generate_batch(args_tuple):
    """生成一批样本的工作函数（用于多进程）"""
    batch_size, seed_offset = args_tuple
    rng = np.random.default_rng(seed_offset)
    
    X_batch = []
    y_batch = []
    neg_stats = dict(VO_drop=0, VI_over50=0)
    
    generated = 0
    # 预留一些余量以防无效样本
    attempts = 0
    max_attempts = batch_size * 2
    
    while generated < batch_size and attempts < max_attempts:
        attempts += 1
        snap = sample_snapshot(rng)
        u, s, neg = schedule_one(snap)
        
        # 若无可调度候选（极少），丢弃该样本，重新采样
        if u is None:
            continue
        
        X_batch.append(snap)
        y_batch.append(u * len(CFG.SERVICES) + s)
        
        # 统计负面事件
        for k, v in neg.items():
            neg_stats[k] += v
        
        generated += 1
    
    return X_batch, y_batch, neg_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="样本数")
    ap.add_argument("--out", type=str, default="data/train.npz", help="输出 npz 路径")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子")
    ap.add_argument("--workers", type=int, default=None, help="并行进程数（默认为CPU核心数）")
    args = ap.parse_args()

    # 确定工作进程数
    n_workers = args.workers if args.workers else cpu_count()
    print(f"使用 {n_workers} 个进程并行生成数据集...")
    
    # 将任务分配给多个进程
    batch_size = max(1, args.n // n_workers)
    tasks = []
    for i in range(n_workers):
        # 每个进程使用不同的种子
        seed_offset = args.seed + i * 10000
        # 最后一个进程处理剩余样本
        if i == n_workers - 1:
            current_batch_size = args.n - (batch_size * (n_workers - 1))
        else:
            current_batch_size = batch_size
        tasks.append((current_batch_size, seed_offset))
    
    # 使用多进程池生成数据
    X = []
    y = []
    neg_stats = dict(VO_drop=0, VI_over50=0)
    
    with Pool(processes=n_workers) as pool:
        # 使用 imap_unordered 来获取进度反馈
        results = []
        with tqdm(total=args.n, ncols=100, desc="Generating") as pbar:
            for result in pool.imap_unordered(generate_batch, tasks):
                X_batch, y_batch, neg_batch = result
                X.extend(X_batch)
                y.extend(y_batch)
                
                # 累计负面事件统计
                for k, v in neg_batch.items():
                    neg_stats[k] += v
                
                pbar.update(len(X_batch))
    
    # 转换为 numpy 数组
    X = np.stack(X, axis=0)         # [N, 128, 3, 2]
    y = np.array(y, dtype=np.int64) # [N]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, meta=dict(
        services=CFG.SERVICES, attrs=CFG.ATTRS, n_users=CFG.N_USERS,
        neg_stats=neg_stats, label_mode="combined"
    ))
    print(f"Saved: {args.out}")
    print("Neg events (sum over dataset):", neg_stats)


if __name__ == "__main__":
    main()
