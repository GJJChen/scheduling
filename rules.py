
# -*- coding: utf-8 -*-
"""传统调度规则（用于打标签）。
输入 snapshot: shape = (N_users, 3, 2)，3 为 VO/VI/BE 顺序，2 为 (buffer_bytes, wait_ms)。
输出: (user_idx, service_idx, neg_events_dict)
"""
from typing import Dict, Tuple, Optional
import numpy as np
from config import CFG

VO, VI, BE = 0, 1, 2
BUF, WAIT = 0, 1

def tx_time_ms(buffer_bytes: float) -> float:
    """基于速率与前导开销估算单次发送时长（毫秒），上限 4ms。"""
    payload_ms = (buffer_bytes * 8.0) / (CFG.RATE_MBPS * 1e3)  # (bytes*8)/(Mbps*1e3) = ms
    t = CFG.PREAMBLE_US / 1000.0 + payload_ms
    return min(t, CFG.MAX_TX_MS)

def schedule_one(snapshot: np.ndarray) -> Tuple[Optional[int], Optional[int], Dict[str, int]]:
    N = snapshot.shape[0]
    buf = snapshot[:, :, BUF]
    wait = snapshot[:, :, WAIT]

    neg = dict(VO_drop=0, VI_over50=0)

    # Step 1: VO 优先；超过 20ms 的 VO 直接丢弃并记负面事件
    vo_mask = buf[:, VO] > 0
    if np.any(vo_mask):
        # 标记过期 VO
        vo_over = (wait[:, VO] > CFG.VO_DROP_MS) & vo_mask
        neg['VO_drop'] += int(np.sum(vo_over))
        # 剔除过期 VO
        vo_valid = vo_mask & (~vo_over)
        if np.any(vo_valid):
            cand_idx = np.argmax(wait[vo_valid, VO])
            # 映射回全局索引
            global_indices = np.nonzero(vo_valid)[0]
            u = int(global_indices[cand_idx])
            return u, VO, neg
        # 若全部过期被丢弃，则继续 Step 2

    # Step 2: VI 若 >20ms 则严格优先。>50ms 记负面事件（仍发送）
    vi_mask = buf[:, VI] > 0
    vi_urgent = vi_mask & (wait[:, VI] > CFG.VI_URGENT_MS)
    if np.any(vi_urgent):
        neg['VI_over50'] += int(np.sum(wait[vi_urgent, VI] > CFG.VI_NEG_MS))
        cand_idx = np.argmax(wait[vi_urgent, VI])
        global_indices = np.nonzero(vi_urgent)[0]
        u = int(global_indices[cand_idx])
        return u, VI, neg

    # Step 3: BE 最小带宽保障（使用 BE.wait_ms 表示距上次服务的时间）
    be_mask = buf[:, BE] > 0
    be_need = be_mask & (wait[:, BE] >= CFG.BE_MIN_PERIOD_MS)
    if np.any(be_need):
        cand_idx = np.argmax(wait[be_need, BE])
        global_indices = np.nonzero(be_need)[0]
        u = int(global_indices[cand_idx])
        return u, BE, neg

    # Step 4: 发送 VI/BE 中“单次发送时长”最长的
    vi_be_mask = np.zeros(N, dtype=bool)
    vi_be_mask |= vi_mask
    vi_be_mask |= be_mask
    if np.any(vi_be_mask):
        # 对每个用户，计算其 VI/BE 的“最长单次发送时长”与对应业务
        best_t = -1.0
        best_u, best_s = None, None
        for u in np.nonzero(vi_be_mask)[0]:
            # 对该用户的 VI/BE 候选分别计算
            for s in (VI, BE):
                if buf[u, s] > 0:
                    t = tx_time_ms(buf[u, s])
                    if t > best_t:
                        best_t, best_u, best_s = t, int(u), int(s)
        if best_u is not None:
            return best_u, best_s, neg

    # 没有任何可发送缓存（按照数据生成逻辑一般不会出现）
    return None, None, neg
