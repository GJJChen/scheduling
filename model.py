# -*- coding: utf-8 -*-
from typing import Optional

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
    """
    使用 Transformer 编码器来拟合调度规则。
    - 置换不变性: 交换输入用户的顺序不影响决策逻辑。
    - 匹配 build_model 接口: 支持两种 num_classes 模式。
    """

    def __init__(self, input_dim=6, num_classes=128,
                 d_model: int = 64, n_head: int = 4,
                 num_encoder_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1):

        super().__init__()
        self.num_classes = num_classes
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
        if num_classes == 128:
            # 模式1: 预测用户
            # Transformer 输出 [B, N, d_model]
            # 我们为每个用户输出 1 个分数
            self.output_head = nn.Linear(d_model, 1)

        elif num_classes % 128 == 0:
            # 模式2: 预测 用户×业务
            # 假设 N=128 (基于你的 MLPClassifier)
            # 推断业务数量 S
            n_services = num_classes // 128
            if n_services == 0:
                raise ValueError(f"num_classes {num_classes} 与 N=128 不兼容")

            # Transformer 输出 [B, N, d_model]
            # 我们为每个用户的 S 个业务输出分数
            self.output_head = nn.Linear(d_model, n_services)

        else:
            raise ValueError(f"num_classes={num_classes} 不被支持 (仅 128 或 128的倍数)")

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # x 形状: (B, N, input_dim), e.g., (64, 128, 6)
        B, N, F = x.shape

        # 1. 投影: (B, N, 6) -> (B, N, d_model)
        x_embed = self.input_projection(x)

        # 2. Transformer: (B, N, d_model) -> (B, N, d_model)
        # self-attention 会在所有 N 个用户之间进行比较
        transformer_out = self.transformer_encoder(
            x_embed,
            src_key_padding_mask=src_key_padding_mask
        )

        # (重要) 处理Padding: 将被掩码的用户的 logits 设置为负无穷
        if src_key_padding_mask is not None:
            # src_key_padding_mask 形状是 (B, N)
            # 我们需要将其扩展到 (B, N, d_model)
            mask_expanded = src_key_padding_mask.unsqueeze(-1).expand_as(transformer_out)
            transformer_out[mask_expanded] = 0.0  # (或者 -1e9, 取决于你)

        # 3. 根据模式输出
        # (B, N, d_model) -> (B, N, 1) 或 (B, N, S)
        logits = self.output_head(transformer_out)

        if self.num_classes == 128:
            # 模式1: 输出 (B, N)
            return logits.squeeze(-1)
        else:
            # 模式2: 输出 (B, N*S), e.g., (B, 384)
            return logits.view(B, -1)

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


class SchedulerTransformer(nn.Module):
    """
    使用Transformer编码器来拟合调度规则的模仿学习模型。

    输入:
        - x (Tensor): 状态快照, 形状 (B, N, 3, 2)
                      B = 批量大小, N = 用户数, 3 = 业务数, 2 = 属性数
        - src_key_padding_mask (Tensor, optional):
                      用于处理变长N的掩码, 形状 (B, N)。
                      True/1 表示该用户是填充的(无效), False/0 表示是真实用户。
    输出:
        - logits (Tensor): 每个可能动作的原始得分, 形状 (B, N*3 + 1)
                           前 N*3 个对应 (u0,s0), (u0,s1)...(uN-1,s2)
                           最后 1 个对应 (None, None) 即 "No-Op" 动作。
    """

    def __init__(self, n_services: int = 3, n_attributes: int = 2,
                 d_model: int = 64, n_head: int = 4,
                 num_encoder_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.n_services = n_services
        self.n_attributes = n_attributes
        self.d_model = d_model

        input_dim = n_services * n_attributes  # 3 * 2 = 6

        # 1. 输入投影层
        # 将每个用户的 (3, 2) 状态展平为 6 维, 再投影到 d_model 维
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Transformer 编码器
        # 注意: batch_first=True, 这样输入形状可以是 (B, N, d_model)
        # 注意: 我们 *不* 使用位置编码, 因为用户顺序是无关的 (置换不变性)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 3. 输出头
        # 将每个用户 (d_model 维) 的输出向量映射到 n_services 个分数
        self.output_head = nn.Linear(d_model, n_services)

        # 4. "No-Op" (不操作) 动作的 Logit
        # 这是一个可学习的参数, 代表选择 "什么都不做" 的基础分数
        self.no_op_logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x 初始形状: (B, N, 3, 2)
        B, N, _, _ = x.shape

        # 1. 展平并投影
        # 形状变为 (B, N, 6)
        x_flat = x.view(B, N, -1)
        # 形状变为 (B, N, d_model)
        x_embed = self.input_projection(x_flat)

        # 2. 通过 Transformer
        # self-attention 会在所有 N 个用户之间进行比较
        # 形状仍为 (B, N, d_model)
        transformer_out = self.transformer_encoder(
            x_embed,
            src_key_padding_mask=src_key_padding_mask
        )

        # 3. 计算每个 (user, service) 动作的 Logits
        # 形状变为 (B, N, 3)
        action_logits = self.output_head(transformer_out)

        # (重要) 处理Padding: 将被掩码的用户的 logits 设置为负无穷
        if src_key_padding_mask is not None:
            # src_key_padding_mask 形状是 (B, N)
            # 我们需要将其扩展为 (B, N, 3)
            mask_expanded = src_key_padding_mask.unsqueeze(-1).expand_as(action_logits)
            action_logits[mask_expanded] = -1e9  # 设为很大的负数

        # 4. 组合所有 Logits
        # 将 (B, N, 3) 展平为 (B, N*3)
        action_logits_flat = action_logits.view(B, -1)

        # 将 "No-Op" logit 扩展到 (B, 1)
        no_op_logit_expanded = self.no_op_logit.expand(B, 1)

        # 最终拼接: (B, N*3 + 1)
        final_logits = torch.cat([action_logits_flat, no_op_logit_expanded], dim=1)

        return final_logits


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
