# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from config import CFG

class BiLSTMClassifier(nn.Module):
    """支持两种模式：
    - num_classes=128: 只预测用户
    - num_classes=384: 预测用户×业务组合
    """
    def __init__(self, input_dim=6, hidden_size=128, num_layers=2, dropout=0.1, num_classes=128):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # *** 关键修复：改变分类头结构 ***
        if num_classes == CFG.N_USERS:
            # User模式: 每个用户输出一个logit
            self.head = nn.Linear(hidden_size * 2, 1)
        else:
            # Combined模式: 每个用户输出3个业务的logit
            self.head = nn.Linear(hidden_size * 2, 3)

    def forward(self, x):
        # x: [B, 128, 6]
        x, _ = self.lstm(x)  # [B, 128, hidden*2]
        
        if self.num_classes == CFG.N_USERS:
            # -> [B, 128, 1] -> [B, 128]
            return self.head(x).squeeze(-1)
        else:
            # -> [B, 128, 3] -> [B, 384]
            x = self.head(x)
            return x.reshape(x.size(0), -1)


class MLPClassifier(nn.Module):
    """支持两种模式：
    - num_classes=128: 只预测用户
    - num_classes=384: 预测用户×业务组合
    """
    def __init__(self, input_dim=6, hidden=256, layers=3, dropout=0.1, num_classes=128):
        super().__init__()
        self.num_classes = num_classes
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
        
        if num_classes == 128:
            blocks += [nn.Linear(hidden, 1)]
            self.per_user = nn.Sequential(*blocks)
            self.final = None
        else:
            self.per_user = nn.Sequential(*blocks)
            # Combined模式：添加最终分类层
            self.final = nn.Sequential(
                nn.Linear(hidden * 128, hidden * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, num_classes)
            )

    def forward(self, x):  # [B, 128, 6]
        B, N, F = x.shape
        x = x.view(B*N, F)
        
        if self.num_classes == 128:
            s = self.per_user(x).view(B, N)
            return s
        else:
            h = self.per_user(x).view(B, N, -1)  # [B, 128, hidden]
            h_flat = h.reshape(B, -1)  # [B, 128*hidden]
            logits = self.final(h_flat)  # [B, num_classes]
            return logits

class TransformerClassifier(nn.Module):
    """支持两种模式：
    - num_classes=128: 只预测用户
    - num_classes=384: 预测用户×业务组合
    """
    def __init__(self, input_dim=6, d_model=128, nhead=8, layers=2, dim_feedforward=256, dropout=0.1, num_classes=128):
        super().__init__()
        self.num_classes = num_classes
        # 输入归一化和投影
        self.inp_norm = nn.LayerNorm(input_dim)
        self.inp = nn.Linear(input_dim, d_model)
        
        # *** 关键修复：添加可学习位置编码 ***
        self.pos_emb = nn.Parameter(torch.randn(1, CFG.N_USERS, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True,
                                                   norm_first=True)  # Pre-LN更稳定
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # *** 关键修复：改变分类头结构 ***
        if num_classes == CFG.N_USERS:
            # User模式: 每个用户输出一个logit
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1)
            )
        else:
            # Combined模式: 每个用户输出3个业务的logit
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 3)
            )

    def forward(self, x):
        # x: [B, 128, 6]
        x = self.inp_norm(x)
        x = self.inp(x)
        x = x + self.pos_emb  # 加上位置编码
        
        x = self.enc(x)  # [B, 128, d_model]
        
        if self.num_classes == CFG.N_USERS:
            # -> [B, 128, 1] -> [B, 128]
            return self.head(x).squeeze(-1)
        else:
            # -> [B, 128, 3] -> [B, 384]
            x = self.head(x)
            return x.reshape(x.size(0), -1)

class HierTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.inp_norm = nn.LayerNorm(input_dim)
        self.inp = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, CFG.N_USERS, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        # 全局汇聚一个 token 用于 service 预测（或用 mean-pool 亦可）
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.service_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        self.user_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))  # 每用户1个logit

    def forward(self, x):  # x: [B, 128, 6]
        x = self.inp_norm(x); x = self.inp(x); x = x + self.pos_emb
        h = self.enc(x)  # [B, 128, d]
        # service logits
        g = self.pool(h.transpose(1,2)).squeeze(-1)  # [B, d]
        service_logits = self.service_head(g)        # [B, 3]
        # user logits（对所有用户给出一个分数，训练时只监督真值 service 下的 argmax 用户）
        user_logits = self.user_head(h).squeeze(-1)  # [B, 128]
        return service_logits, user_logits


def build_model(name: str, input_dim=6, num_classes=128):
    """构建模型
    Args:
        name: 模型名称 (bilstm, mlp, transformer)
        input_dim: 输入特征维度
        num_classes: 输出类别数 (128=仅用户, 384=用户×业务)
    """
    name = name.lower()
    if name == "bilstm":
        return BiLSTMClassifier(input_dim=input_dim, num_classes=num_classes)
    if name == "mlp":
        return MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    if name == "transformer":
        return TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
