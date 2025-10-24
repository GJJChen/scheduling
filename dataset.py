
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from config import CFG

class SchedulingNPZDataset(Dataset):
    """从 npz 加载 [N, 128, 3, 2] -> 每样本返回 (features [128, 6], label int)。"""
    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]        # [N, 128, 3, 2]
        self.y = data["y"]        # [N]
        # 转换为 float32/int64 的 PyTorch 张量时再处理

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        snap = self.X[idx]                     # [128, 3, 2]
        feats = snap.reshape(CFG.N_USERS, -1)  # [128, 6]  (VO/VI/BE × (buffer, wait))
        label = int(self.y[idx])
        return torch.from_numpy(feats).float(), torch.tensor(label, dtype=torch.long)

def split_dataset(full: Dataset, val_ratio=0.1, seed=42):
    n = len(full)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(full, [n_train, n_val], generator=gen)
