
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

class SchedulingNPZDataset(Dataset):
    """从 npz 加载 [N, n_users, 3, 2] -> 每样本返回 (features [n_users, 6], label int)
    初始化时就归一化，加速后面的训练
    两种标签模式：'user' (0-n_users-1) 或 'combined' (0-n_users*3-1)
    """
    def __init__(self, npz_path: str, normalize=True, precompute=True, norm_params=None):
        """
        Args:
            npz_path: npz文件路径
            normalize: 是否归一化
            precompute: 是否预计算归一化
            norm_params: 外部归一化参数字典，包含 'buf_max' 和 'wait_max'。
                        如果提供，将使用这些参数而不是从当前数据计算
        """
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]        # [N, n_users, 3, 2]
        self.y = data["y"]        # [N]
        self.normalize = normalize
        self.precompute = precompute
        self.norm_params = norm_params
        
        # 从数据形状推断用户数
        self.n_users = self.X.shape[1]
        self.n_services = self.X.shape[2]
        
        # 读取标签模式（如果存在）
        if "meta" in data:
            meta = data["meta"].item()
            self.label_mode = meta.get("label_mode", "user")
            self.num_classes = self.n_users * self.n_services if self.label_mode == "combined" else self.n_users
        else:
            # user 模式
            self.label_mode = "user"
            self.num_classes = self.n_users
        
        # 预计算归一化（加速训练）
        if self.normalize and self.precompute:
            print(f"预计算 归一化数据... 数据集大小: {self.X.shape}")
            
            # 如果提供了外部归一化参数，使用它们；否则从当前数据计算
            if self.norm_params is not None:
                buf_max = self.norm_params['buf_max']
                wait_max = self.norm_params['wait_max']
                print(f"使用外部归一化参数: buf_max={buf_max}, wait_max={wait_max}")
            else:
                # buffer_bytes 归一化 (VO/VI/BE分别计算最大值)
                # X shape: [N, 128, 3, 2] - 维度2是服务类型(VO/VI/BE)，维度3是属性(buffer/wait)
                buf_max = np.array([
                    self.X[:, :, 0, 0].max(),  # VO buffer max
                    self.X[:, :, 1, 0].max(),  # VI buffer max
                    self.X[:, :, 2, 0].max(),  # BE buffer max
                ], dtype=np.float32)
                
                # wait_ms 归一化 (VO/VI/BE分别计算最大值)
                wait_max = np.array([
                    self.X[:, :, 0, 1].max(),  # VO wait max
                    self.X[:, :, 1, 1].max(),  # VI wait max
                    self.X[:, :, 2, 1].max(),  # BE wait max
                ], dtype=np.float32)
            
            # 存储归一化参数
            self.buf_max = buf_max
            self.wait_max = wait_max
            
            # 预归一化整个数据集
            self.X = self.X.astype(np.float32)  # 确保是float32
            for s in range(3):
                if self.buf_max[s] > 0:
                    self.X[:, :, s, 0] = self.X[:, :, s, 0] / self.buf_max[s]
                if self.wait_max[s] > 0:
                    self.X[:, :, s, 1] = self.X[:, :, s, 1] / self.wait_max[s]
            
            # 预reshape为 [N, n_users, 6] 格式
            self.X = self.X.reshape(self.X.shape[0], self.n_users, -1)
            print(f"归一化完成，数据已reshape为: {self.X.shape}")
            
        elif self.normalize and not self.precompute:
            # 如果不预计算，存储归一化参数
            if self.norm_params is not None:
                self.buf_max = self.norm_params['buf_max']
                self.wait_max = self.norm_params['wait_max']
            else:
                self.buf_max = np.array([
                    self.X[:, :, 0, 0].max(),
                    self.X[:, :, 1, 0].max(),
                    self.X[:, :, 2, 0].max(),
                ], dtype=np.float32)
                self.wait_max = np.array([
                    self.X[:, :, 0, 1].max(),
                    self.X[:, :, 1, 1].max(),
                    self.X[:, :, 2, 1].max(),
                ], dtype=np.float32)
    
    def get_norm_params(self):
        """获取归一化参数"""
        if self.normalize and hasattr(self, 'buf_max') and hasattr(self, 'wait_max'):
            return {
                'buf_max': self.buf_max,
                'wait_max': self.wait_max
            }
        return None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.normalize and self.precompute:
            # 已经预归一化和reshape，直接返回
            feats = self.X[idx]  # [n_users, 6]
        else:
            snap = self.X[idx].copy()  # [n_users, 3, 2]
            
            if self.normalize:
                # 运行时归一化（慢）
                for s in range(3):
                    if self.buf_max[s] > 0:
                        snap[:, s, 0] = snap[:, s, 0] / self.buf_max[s]
                    if self.wait_max[s] > 0:
                        snap[:, s, 1] = snap[:, s, 1] / self.wait_max[s]
            
            feats = snap.reshape(self.n_users, -1)  # [n_users, 6]
        
        label = int(self.y[idx])
        return torch.from_numpy(feats).float(), torch.tensor(label, dtype=torch.long)

def split_dataset(full: Dataset, val_ratio=0.1, seed=42):
    n = len(full)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(full, [n_train, n_val], generator=gen)
