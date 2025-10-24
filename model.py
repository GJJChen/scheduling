
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from config import CFG

class BiLSTMClassifier(nn.Module):
    """按“用户维度”作为序列建模，输出每个用户的打分，softmax 做用户分类。"""
    def __init__(self, input_dim=6, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0,
                            bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):  # x: [B, 128, 6]
        h, _ = self.lstm(x)          # [B, 128, 2H]
        logits = self.head(h).squeeze(-1)  # [B, 128]
        return logits

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden=256, layers=3, dropout=0.1):
        super().__init__()
        blocks = []
        d = input_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        blocks += [nn.Linear(hidden, 1)]
        self.per_user = nn.Sequential(*blocks)

    def forward(self, x):  # [B, 128, 6]
        B, N, F = x.shape
        x = x.view(B*N, F)
        s = self.per_user(x).view(B, N)
        return s

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x):  # [B, 128, 6]
        h = self.inp(x)           # [B, 128, d_model]
        h = self.enc(h)           # [B, 128, d_model]
        logits = self.head(h).squeeze(-1)  # [B, 128]
        return logits

def build_model(name: str, input_dim=6):
    name = name.lower()
    if name == "bilstm":
        return BiLSTMClassifier(input_dim=input_dim)
    if name == "mlp":
        return MLPClassifier(input_dim=input_dim)
    if name == "transformer":
        return TransformerClassifier(input_dim=input_dim)
    raise ValueError(f"Unknown model: {name}")
