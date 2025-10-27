
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from config import CFG

class SchedulingNPZDataset(Dataset):
    """从 npz 加载 [N, 128, 3, 2] -> 每样本返回 (features [128, 6], label int)。
    添加特征归一化以提高训练稳定性。
    """
    def __init__(self, npz_path: str, normalize=True):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]        # [N, 128, 3, 2]
        self.y = data["y"]        # [N]
        self.normalize = normalize
        
        # 计算归一化参数（基于整个数据集）
        if self.normalize:
            # buffer_bytes 归一化 (VO/VI/BE分别归一化)
            self.buf_max = np.array([
                CFG.BUF_RANGE['VO'][1],  # VO max
                CFG.BUF_RANGE['VI'][1],  # VI max
                CFG.BUF_RANGE['BE'][1],  # BE max
            ], dtype=np.float32)
            
            # wait_ms 归一化 (VO/VI/BE分别归一化)
            self.wait_max = np.array([
                CFG.WAIT_RANGE['VO'][1],  # VO max
                CFG.WAIT_RANGE['VI'][1],  # VI max
                CFG.WAIT_RANGE['BE'][1],  # BE max
            ], dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        snap = self.X[idx].copy()  # [128, 3, 2]
        
        if self.normalize:
            # 归一化 buffer_bytes: [128, 3, 0]
            for s in range(3):
                if self.buf_max[s] > 0:
                    snap[:, s, 0] = snap[:, s, 0] / self.buf_max[s]
            
            # 归一化 wait_ms: [128, 3, 1]
            for s in range(3):
                if self.wait_max[s] > 0:
                    snap[:, s, 1] = snap[:, s, 1] / self.wait_max[s]
        
        feats = snap.reshape(CFG.N_USERS, -1)  # [128, 6]  (VO/VI/BE × (buffer, wait))
        label = int(self.y[idx])
        return torch.from_numpy(feats).float(), torch.tensor(label, dtype=torch.long)

def split_dataset(full: Dataset, val_ratio=0.1, seed=42):
    n = len(full)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(full, [n_train, n_val], generator=gen)
