
# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # 基本规模
    N_USERS: int = 128
    SERVICES: tuple = ("VO", "VI", "BE")  # 按固定顺序 0/1/2
    ATTRS: tuple = ("buffer_bytes", "wait_ms")

    # 物理层参数
    RATE_MBPS: float = 2442.0         # 双流 MCS11
    PREAMBLE_US: float = 150.0        # EDCA+Preamble 固定开销
    MAX_TX_MS: float = 4.0

    # 时延门限
    VO_DROP_MS: float = 20.0          # VO 超过即丢弃并记负面事件
    VI_URGENT_MS: float = 20.0        # VI 超过即严格优先
    VI_NEG_MS: float = 50.0           # VI 超过记负面事件（仍发送）
    BE_MIN_PERIOD_MS: float = 100.0   # BE 最小保障：每隔 100ms 至少一次机会（用 wait_ms 表示间隔）

    # 随机采样范围（可按需调整）
    P_HAS_BUF = dict(VO=0.25, VI=0.35, BE=0.55)  # 各业务出现非零缓存的概率
    # 业务报文大小分布（字节）
    BUF_RANGE = dict(VO=(200, 4000), VI=(400, 20000), BE=(400, 1500000))
    # 等待时延采样范围（毫秒）
    WAIT_RANGE = dict(VO=(0.0, 40.0), VI=(0.0, 80.0), BE=(0.0, 150.0))

CFG = Config()
