
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
from tqdm import tqdm
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="样本数")
    ap.add_argument("--out", type=str, default="data/train.npz", help="输出 npz 路径")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = []
    y = []
    neg_stats = dict(VO_drop=0, VI_over50=0)

    pbar = tqdm(total=args.n, ncols=100, desc="Generating")
    while len(X) < args.n:
        snap = sample_snapshot(rng)
        u, s, neg = schedule_one(snap)

        # 若无可调度候选（极少），丢弃该样本，重新采样
        if u is None:
            continue

        # 记录样本与标签（只存“调度哪个用户”用于监督学习）
        X.append(snap)
        y.append(u)

        # 统计负面事件（仅记录以供观测）
        for k, v in neg.items():
            neg_stats[k] += v

        pbar.update(1)
    pbar.close()

    X = np.stack(X, axis=0)         # [N, 128, 3, 2]
    y = np.array(y, dtype=np.int64) # [N]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, meta=dict(
        services=CFG.SERVICES, attrs=CFG.ATTRS, n_users=CFG.N_USERS,
        neg_stats=neg_stats
    ))
    print(f"Saved: {args.out}")
    print("Neg events (sum over dataset):", neg_stats)

if __name__ == "__main__":
    main()
