# -*- coding: utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """两种模式：
    - num_classes=n_users: 只预测用户
    - num_classes=n_users*n_services: 预测用户×业务组合
    """
    def __init__(self, input_dim=6, hidden_size=128, num_layers=2, dropout=0.1, num_classes=128, n_users=128):
        super().__init__()
        self.num_classes = num_classes
        self.n_users = n_users
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # *** 关键修复：改变分类头结构 ***
        if num_classes == n_users:
            # User模式: 每个用户输出一个logit
            self.head = nn.Linear(hidden_size * 2, 1)
        else:
            # Combined模式: 每个用户输出3个业务的logit
            self.head = nn.Linear(hidden_size * 2, 3)

    def forward(self, x):
        # x: [B, n_users, 6]
        x, _ = self.lstm(x)  # [B, n_users, hidden*2]
        
        if self.num_classes == self.n_users:
            # -> [B, n_users, 1] -> [B, n_users]
            return self.head(x).squeeze(-1)
        else:
            # -> [B, n_users, 3] -> [B, n_users*3]
            x = self.head(x)
            return x.reshape(x.size(0), -1)


class MLPClassifier(nn.Module):
    """两种模式：
    - num_classes=n_users: 只预测用户
    - num_classes=n_users*n_services: 预测用户×业务组合
    """
    def __init__(self, input_dim=6, hidden=256, layers=3, dropout=0.1, num_classes=128, n_users=128):
        super().__init__()
        self.num_classes = num_classes
        self.n_users = n_users
        blocks = []
        # 输入归一化
        blocks.append(nn.LayerNorm(input_dim))
        d = input_dim
        for i in range(layers):
            blocks += [
                nn.Linear(d, hidden),
                nn.LayerNorm(hidden),  # 添加LayerNorm
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            d = hidden
        
        if num_classes == n_users:
            blocks += [nn.Linear(hidden, 1)]
            self.per_user = nn.Sequential(*blocks)
            self.final = None
        else:
            self.per_user = nn.Sequential(*blocks)
            # Combined模式：添加最终分类层
            self.final = nn.Sequential(
                nn.Linear(hidden * n_users, hidden * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, num_classes)
            )

    def forward(self, x):  # [B, n_users, 6]
        B, N, F = x.shape
        x = x.view(B*N, F)
        
        if self.num_classes == self.n_users:
            s = self.per_user(x).view(B, N)
            return s
        else:
            h = self.per_user(x).view(B, N, -1)  # [B, n_users, hidden]
            h_flat = h.reshape(B, -1)  # [B, n_users*hidden]
            logits = self.final(h_flat)  # [B, num_classes]
            return logits


class TransformerClassifier(nn.Module):
    """
    使用 Transformer 编码器来拟合调度规则。
    - 置换不变性: 交换输入用户的顺序不影响决策逻辑。
    - 匹配 build_model 接口: 支持两种 num_classes 模式。
    """

    def __init__(self, input_dim=6, num_classes=128, n_users=128,
                 d_model: int = 64, n_head: int = 4,
                 num_encoder_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1):

        super().__init__()
        self.num_classes = num_classes
        self.n_users = n_users
        self.d_model = d_model

        # 1. 输入投影层
        # 将每个用户的 6 维特征投影到 d_model 维
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Transformer 编码器
        # batch_first=True 使得输入形状为 (B, N, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 3. 输出头 (根据 num_classes 变化)
        if num_classes == n_users:
            # 模式1: 预测用户
            # Transformer 输出 [B, N, d_model]
            # 我们为每个用户输出 1 个分数
            self.output_head = nn.Linear(d_model, 1)

        elif num_classes % n_users == 0:
            # 模式2: 预测 用户×业务
            # 推断业务数量 S
            n_services = num_classes // n_users
            if n_services == 0:
                raise ValueError(f"num_classes {num_classes} 与 n_users={n_users} 不兼容")

            # Transformer 输出 [B, N, d_model]
            # 为每个用户的 S 个业务输出分数
            self.output_head = nn.Linear(d_model, n_services)

        else:
            raise ValueError(f"num_classes={num_classes} 不被支持 (仅 n_users 或 n_users的倍数)")

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # x 形状: (B, N_users, input_dim)
        B, N, F = x.shape

        # 1. 投影: (B, N, 6) -> (B, N, d_model)
        x_embed = self.input_projection(x)

        # 2. Transformer: (B, N, d_model) -> (B, N, d_model)
        # self-attention 会在所有 N 个用户之间进行比较
        transformer_out = self.transformer_encoder(
            x_embed,
            src_key_padding_mask=src_key_padding_mask
        )

        # Padding: 将被掩码的用户的 logits 设置为负无穷
        if src_key_padding_mask is not None:
            # src_key_padding_mask 形状是 (B, N)
            # 需将其扩展到 (B, N, d_model)
            mask_expanded = src_key_padding_mask.unsqueeze(-1).expand_as(transformer_out)
            transformer_out[mask_expanded] = 0.0

        # 3. 根据模式输出
        # (B, N, d_model) -> (B, N, 1) 或 (B, N, S)
        logits = self.output_head(transformer_out)

        if self.num_classes == self.n_users:
            # 模式1: 输出 (B, N)
            return logits.squeeze(-1)
        else:
            # 模式2: 输出 (B, N*S), e.g., (B, n_users*n_services)
            return logits.view(B, -1)

def build_model(name: str, input_dim=6, num_classes=128, n_users=128):
    """构建模型
    Args:
        name: 模型名称 (bilstm, mlp, transformer)
        input_dim: 输入特征维度
        num_classes: 输出类别数 (n_users=仅用户, n_users*n_services=用户×业务)
        n_users: 用户数量
    """
    name = name.lower()
    if name == "bilstm":
        return BiLSTMClassifier(input_dim=input_dim, num_classes=num_classes, n_users=n_users)
    if name == "mlp":
        return MLPClassifier(input_dim=input_dim, num_classes=num_classes, n_users=n_users)
    if name == "transformer":
        return TransformerClassifier(input_dim=input_dim, num_classes=num_classes, n_users=n_users)

    raise ValueError(f"Unknown model: {name}")
