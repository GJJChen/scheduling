import random
from functools import cmp_to_key
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import threading
import time
import multiprocessing
from multiprocessing import Manager
import cProfile
import heapq
import math
import argparse # 导入 argparse
from tqdm import tqdm # 导入 tqdm

# --- 强化学习所需的库 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym # 使用 Gymnasium (OpenAI Gym 的继任者)
from gymnasium import spaces

# ##################################################################
# --- (开始) 您的原始仿真代码 (未修改) ---
# ##################################################################

class Task:
    def __init__(self, event, timestamp):
        self.event = event
        self.timestamp = timestamp
    def __lt__(self, other):
        # 定义小于号的比较方式，用于 heapq 排序
        return self.timestamp < other.timestamp
    def __repr__(self):
        # 修复一个小的 repr 错误 (name -> event, priority 不存在)
        return f"Task(event={self.event}, timestamp={self.timestamp})"

def compare(a, b):
    if a["rate"] < b["rate"]:
        return 1
    return -1

def cmp(a, b):
    if a["timestamp"] > b["timestamp"]:
        return 1
    elif a["timestamp"] == b["timestamp"]:
        if a["size"] < b["size"]:
            return 1
        else:
            return -1
    else:
       return -1

def create_txq_table(user_num, queue_size):
    txq_table = []
    # 创建一个线程安全的队列
    for user_id in range(user_num):
        tid_table = []
        for tid in range(3):
            #这里使用环形队列实现
            tid_table.append([0 for _ in range(queue_size)])
        txq_table.append(tid_table)
    return txq_table

def create_tx_info_table(user_num):
    tx_info_table = []
    # 创建一个线程安全的队列
    for user_id in range(user_num):
        tx_info_tid_table = []
        for tid in range(3):
            tx_info_tid_table.append({"tot_size" : 0, "read_ptr" : 0, "write_ptr" : 0})
        tx_info_table.append(tx_info_tid_table)
    return tx_info_table
   
def create_rate_table(user_num):
    rate_table = []
    max_band_width = 2442
    ofo_list = []
    for user_id in range(user_num):
        tid_rate_table = []
        for tid in range(3):
            if tid == 0:
                rand_rate = random.randint(1, 20)
            elif tid == 1:
                rand_rate = random.randint(1, 40)
            elif tid == 2:
                rand_rate = random.randint(1, 100)
            #ratetable包含两个信息，速率和发包间隔
            #tid_rate_table.append(rand_rate * 1000 * 1000)
            if tid == 0:
                rand_interval = random.randint(1, 20)
                #tid_rate_table.append(rand_interval)
            elif tid == 1:
                rand_interval = random.randint(1, 10)
                #tid_rate_table.append(rand_interval)
            else:
                rand_interval = random.randint(1, 3)
                #tid_rate_table.append(rand_interval)
            rate_info = {
                'uid' : user_id,
                'tid' : tid,
                'rate' : rand_rate
            }
            tid_rate_table.append([0, rand_interval])
            ofo_list.append(rate_info)
        #速率先写入0
        rate_table.append(tid_rate_table)
    sorted_list = sorted(ofo_list, key=cmp_to_key(compare))
   
    sum_rate = 0
    for i in range(len(sorted_list)):
        if sum_rate + sorted_list[i]['rate'] < max_band_width:
            uid = sorted_list[i]['uid']
            tid = sorted_list[i]['tid']
            rate_table[uid][tid][0] = sorted_list[i]['rate'] * 1000 * 1000
            #print(f"user {uid}, ac {tid}, set rate is {sorted_list[i]['rate']}")
            sum_rate = sum_rate + sorted_list[i]['rate']
    #调试用
    '''
    for i in range(user_num):
        for j in range(3):
            print(f"user {i}, ac {j}, rate is {rate_table[i][j][0]}")
    '''
    return rate_table

#
# 传统调度器核心逻辑 (PPO 环境也会复用其中的 丢包逻辑)
#
def gen_sched_res(txq_table, tx_info_table, user_num, timestamp_now, queue_size, stats_dict=None):
    vo_list = []
    vi_list = []
    be_list = []
    #首先从txq_table中取首包做排序
    schedule_res = {"uid" : -1, "ac_type" : -1}
    
    # --- PPO 环境将复用此丢包逻辑 ---
    # 遍历所有队列，检查并丢弃超时报文
    for uid in range(user_num):
        # VO 队列 (ac=0)
        r_ptr_vo = tx_info_table[uid][0]["read_ptr"]
        w_ptr_vo = tx_info_table[uid][0]["write_ptr"]
        while r_ptr_vo != w_ptr_vo:
            if timestamp_now - txq_table[uid][0][r_ptr_vo]['timestamp'] > 20:
                # 记录丢弃长度
                # print(f"Traditional Drop VO: uid {uid} timestamp {timestamp_now} pkt_ts {txq_table[uid][0][r_ptr_vo]['timestamp']}")
                pkt_size = txq_table[uid][0][r_ptr_vo]['size']
                tx_info_table[uid][0]["tot_size"] = tx_info_table[uid][0]["tot_size"] - pkt_size
                if stats_dict is not None:
                    stats_dict['drops_timeout'] += 1
                    stats_dict['drops_timeout_bytes'] += pkt_size
                r_ptr_vo = (r_ptr_vo + 1) % queue_size
            else:
                break # 队列是 FIFO 的，第一个没超时，后面的也不会
        tx_info_table[uid][0]["read_ptr"] = r_ptr_vo # 更新读指针

        # VI 队列 (ac=1)
        r_ptr_vi = tx_info_table[uid][1]["read_ptr"]
        w_ptr_vi = tx_info_table[uid][1]["write_ptr"]
        while r_ptr_vi != w_ptr_vi:
            if timestamp_now - txq_table[uid][1][r_ptr_vi]['timestamp'] > 50:
                # 记录丢弃长度
                # print(f"Traditional Drop VI: uid {uid} timestamp {timestamp_now} pkt_ts {txq_table[uid][1][r_ptr_vi]['timestamp']}")
                pkt_size = txq_table[uid][1][r_ptr_vi]['size']
                tx_info_table[uid][1]["tot_size"] = tx_info_table[uid][1]["tot_size"] - pkt_size
                if stats_dict is not None:
                    stats_dict['drops_timeout'] += 1
                    stats_dict['drops_timeout_bytes'] += pkt_size
                r_ptr_vi = (r_ptr_vi + 1) % queue_size
            else:
                break
        tx_info_table[uid][1]["read_ptr"] = r_ptr_vi
    # --- 丢包逻辑结束 ---

    # 丢包后，重新收集所有非空队列的包头信息
    for uid in range(user_num):
        for ac in range(3):
            w_ptr = tx_info_table[uid][ac]["write_ptr"]
            r_ptr = tx_info_table[uid][ac]["read_ptr"]
            if w_ptr != r_ptr: #队列非空
                pkt_info = txq_table[uid][ac][r_ptr]
                schedule_info = {
                    "uid" : uid,
                    "ac_type" : ac,
                    "size" : tx_info_table[uid][ac]["tot_size"],
                    "timestamp" : pkt_info['timestamp'] # 包头时间戳
                }
                if ac == 0:
                    vo_list.append(schedule_info)
                elif ac == 1:
                    vi_list.append(schedule_info)
                else:
                    be_list.append(schedule_info)
               
    if len(vo_list) == 0 and len(vi_list) == 0 and len(be_list) == 0:
        schedule_res = {"uid" : -1, "ac_type" : -1}
        return schedule_res
    
    sorted_vo_list = sorted(vo_list, key=cmp_to_key(cmp))
    sorted_vi_list = sorted(vi_list, key=cmp_to_key(cmp))
    sorted_be_list = sorted(be_list, key=cmp_to_key(cmp))

    # --- 传统调度规则开始 ---
    #尝试调度VO
    for i in range(len(sorted_vo_list)):
        # 规则：时延在 (0, 20ms] 之间
        if 0 < timestamp_now - sorted_vo_list[i]["timestamp"] <= 20:
            schedule_res = {"uid" : sorted_vo_list[i]['uid'], "ac_type" : 0}
            return schedule_res
        # > 20ms 的已经在上面被丢弃了
            
    #尝试调度VI
    for i in range(len(sorted_vi_list)):
         # 规则：时延在 (20, 50ms] 之间
        if 20 < timestamp_now - sorted_vi_list[i]["timestamp"] <= 50:
            schedule_res = {"uid" : sorted_vi_list[i]['uid'], "ac_type" : 1}
            return schedule_res
        # > 50ms 的已经在上面被丢弃了

    #调度BE，假定所有用户的最小需求都是100ms调度一次
    for i in range(len(sorted_be_list)):
        if timestamp_now - sorted_be_list[i]["timestamp"] > 100:
            schedule_res = {"uid" : sorted_be_list[i]['uid'], "ac_type" : 2}
            return schedule_res
            
    #没有调度到BE，调度剩余报文最长的用户 (VI 和 BE 中)
    max_pkt_size = 0
    res_uid = -1
    res_ac_type = -1

    # 遍历 VI 队列
    for vi_info in sorted_vi_list:
        if vi_info["size"] > max_pkt_size:
            max_pkt_size = vi_info["size"]
            res_uid = vi_info["uid"]
            res_ac_type = vi_info["ac_type"] # 1

    # 遍历 BE 队列
    for be_info in sorted_be_list:
         if be_info["size"] > max_pkt_size:
            max_pkt_size = be_info["size"]
            res_uid = be_info["uid"]
            res_ac_type = be_info["ac_type"] # 2

    # 如果 vo, vi, be 队列都为空, res_uid 会是 -1, 此时 gen_sched_res 已在开头返回
    # 如果 vo 队列不空, 但不满足 (0, 20] 条件, 它也会在这里被选中 (如果 BE/VI 也为空)
    # 这是一个潜在的规则空隙，但我们保持原样
    if res_uid == -1:
        # 此时 BE/VI 必为空，VO 队列可能不空但未满足 (0, 20]
        if len(sorted_vo_list) > 0:
            # 按原逻辑，vo_list 不参与 "最长队列" 比较，但如果其他都空，只能选它
            # 我们选择最早的 VO 包
            res_uid = sorted_vo_list[0]['uid']
            res_ac_type = sorted_vo_list[0]['ac_type'] # 0
        else:
            # 应该在函数开头就返回了
             schedule_res = {"uid" : -1, "ac_type" : -1}
             return schedule_res

    schedule_res = {"uid" : res_uid, "ac_type" : res_ac_type}
    return schedule_res
   
def schedule(txq_table, tx_info_table, user_num, timestamp, queue_size, event_queue, stats_dict=None):
    max_rate = 2442 * 1000 * 1000
    
    # *** 这是传统调度器的决策点 ***
    sched_res = gen_sched_res(txq_table, tx_info_table, user_num, timestamp, queue_size, stats_dict)

    sched_uid = sched_res["uid"]
    sched_tid = sched_res["ac_type"]
    
    if sched_uid == -1:
        # print(f"nothing to schedule!!")
        heapq.heappush(event_queue, Task("schedule", timestamp + 1)) # 1ms 后再看
        return
        
    # print(f"schedule uid is {sched_uid}, tid is {sched_tid}")
    temp_sched_size = 0
    r_ptr = tx_info_table[sched_uid][sched_tid]["read_ptr"]
    w_ptr = tx_info_table[sched_uid][sched_tid]["write_ptr"]
    
    max_sched_bytes = max_rate / 1000 * 4 
    
    sched_start_r_ptr = r_ptr # 记录开始的读指针

    while r_ptr != w_ptr:
        if temp_sched_size == max_sched_bytes:
            break
            
        pkt_size = txq_table[sched_uid][sched_tid][r_ptr]["size"]
        
        if temp_sched_size + pkt_size <= max_sched_bytes:
            if stats_dict is not None:
                pkt_timestamp = txq_table[sched_uid][sched_tid][r_ptr]["timestamp"]
                stats_dict['latency_sum'] += (timestamp - pkt_timestamp)
                stats_dict['packets_sent'] += 1
            temp_sched_size = temp_sched_size + pkt_size
            r_ptr = (r_ptr + 1) % queue_size
        else:
            # 报文切分
            remaining_bytes = max_sched_bytes - temp_sched_size
            if stats_dict is not None:
                pkt_timestamp = txq_table[sched_uid][sched_tid][r_ptr]["timestamp"]
                # 按比例计算切片时延
                stats_dict['latency_sum'] += (timestamp - pkt_timestamp) * (remaining_bytes / pkt_size)
                # 不算一个完整的包
            txq_table[sched_uid][sched_tid][r_ptr]["size"] = pkt_size - remaining_bytes
            temp_sched_size = max_sched_bytes
            # r_ptr 不动，下次从这个被切分的包开始
            break
            
    tx_info_table[sched_uid][sched_tid]["read_ptr"] = r_ptr
    tx_info_table[sched_uid][sched_tid]["tot_size"] = tx_info_table[sched_uid][sched_tid]["tot_size"] - temp_sched_size
    if stats_dict is not None:
        stats_dict['bytes_sent'] += temp_sched_size
    
    # 保证 send_time > 0，避免事件循环卡住
    send_time = max(temp_sched_size / max_rate * 1000, 0.001) # 至少推进 0.001 ms
    
    # print(f"schedule complete timestamp is {timestamp}, send_time is {send_time}, temp_sched_size is {temp_sched_size}, queue remain size is {tx_info_table[sched_uid][sched_tid]['tot_size']}")
   
    heapq.heappush(event_queue, Task("schedule", timestamp + send_time))

def traffic_generator(txq_table, user_num, timestamp, rate_table, tx_info_table, queue_size, event_queue, stats_dict=None):
    packets_dropped_full = 0 # 为 PPO 环境增加返回值
    for uid in range(user_num):
        for ac_num in range(3):
            if rate_table[uid][ac_num][0] > 0:
                #速率表中指定了发包间隔
                if timestamp % rate_table[uid][ac_num][1] == 0:
                    if (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size == tx_info_table[uid][ac_num]["read_ptr"]:
                        # print(f"time {timestamp} uid {uid}, tid {ac_num} queue is full, drop input packet")
                        packets_dropped_full += 1
                        if stats_dict is not None:
                            stats_dict['drops_full'] += 1
                        continue
                    packet_size = np.random.poisson(lam=rate_table[uid][ac_num][0] * 0.001)
                    if packet_size == 0: continue # 不发送 0 字节包
                    
                    pkt_info = {}
                    pkt_info['size'] = packet_size
                    pkt_info['timestamp'] = timestamp
                    
                    tx_info_table[uid][ac_num]["tot_size"] = tx_info_table[uid][ac_num]["tot_size"] + packet_size
                    w_ptr = tx_info_table[uid][ac_num]["write_ptr"]
                    txq_table[uid][ac_num][w_ptr] = pkt_info
                    tx_info_table[uid][ac_num]["write_ptr"] = (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size
                    
    heapq.heappush(event_queue, Task("traffic_generate", timestamp + 1))
    return packets_dropped_full # PPO 环境需要这个统计

# ##################################################################
# --- (结束) 您的原始仿真代码 ---
# ##################################################################


# ##################################################################
# --- (开始) PPO 及 Gym 环境整合代码 ---
# ##################################################################

# --- 1. PPO Agent 的神经网络定义 ---

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """初始化神经网络层"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    """Actor 网络 (策略网络)"""
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, x, action=None):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

class CriticNetwork(nn.Module):
    """Critic 网络 (价值网络)"""
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(self, x):
        return self.net(x)

    def get_value(self, x):
        return self.forward(x)

# --- 2. Gym 环境封装 ---

class SimulationEnv(gym.Env):
    """
    将您的仿真逻辑封装为 Gym 环境
    """
    metadata = {"render_modes": []}

    def __init__(self, user_num, queue_size, episode_length=10000):
        super(SimulationEnv, self).__init__()
        
        self.user_num = user_num
        self.queue_size = queue_size
        self.num_acs = 3
        self.episode_length = episode_length # ms
        self.max_rate = 2442 * 1000 * 1000 # bits/sec
        self.max_sched_bytes = self.max_rate / 1000 * 4 # 同原 schedule 逻辑

        # 状态 (Observation) 空间:
        # 每个队列 (user * ac) 有 3 个特征:
        # 1. tot_size (队列总字节数)
        # 2. hol_delay (Head-of-Line, 包头时延)
        # 3. packet_count (队列中包的数量)
        self.n_features_per_queue = 3
        state_dim = self.user_num * self.num_acs * self.n_features_per_queue
        
        # 动作 (Action) 空间:
        # (user_num * num_acs) 个动作, 对应调度每个队列
        # +1 个 "no-op" 动作 (当所有队列都为空时)
        self.action_dim = self.user_num * self.num_acs + 1
        self.no_op_action = 0 # 动作 0 定义为 no-op
        
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # 仿真内部状态
        self.txq_table = None
        self.tx_info_table = None
        self.rate_table = None
        self.event_queue = None
        self.timestamp_now = 0.0

        # PPO 统计
        self.total_bytes_sent = 0
        self.total_packets_sent = 0
        self.total_latency = 0
        self.total_drops_full = 0
        self.total_drops_timeout = 0

    def _get_obs(self):
        """从仿真状态生成 PPO 的观测向量"""
        obs = np.zeros((self.user_num, self.num_acs, self.n_features_per_queue), dtype=np.float32)
        
        for u in range(self.user_num):
            for a in range(self.num_acs):
                info = self.tx_info_table[u][a]
                r_ptr = info["read_ptr"]
                w_ptr = info["write_ptr"]
                
                tot_size = info["tot_size"]
                packet_count = (w_ptr - r_ptr + self.queue_size) % self.queue_size
                hol_delay = 0.0
                
                if packet_count > 0:
                    hol_timestamp = self.txq_table[u][a][r_ptr]["timestamp"]
                    hol_delay = self.timestamp_now - hol_timestamp
                
                # 状态归一化 (简单处理)
                obs[u, a, 0] = tot_size / 1e6 # 假设 1MB 是大的队列
                obs[u, a, 1] = hol_delay / 100.0 # 假设 100ms 是大时延
                obs[u, a, 2] = packet_count / self.queue_size
                
        return obs.flatten()

    def _discard_old_packets(self):
        """
        PPO 环境中独立的丢包逻辑 (复用 gen_sched_res 的逻辑)
        返回: (丢弃的包数, 丢弃的字节数)
        """
        packets_dropped = 0
        bytes_dropped = 0

        for uid in range(self.user_num):
            # VO 队列 (ac=0), 20ms
            r_ptr_vo = self.tx_info_table[uid][0]["read_ptr"]
            w_ptr_vo = self.tx_info_table[uid][0]["write_ptr"]
            while r_ptr_vo != w_ptr_vo:
                pkt_timestamp = self.txq_table[uid][0][r_ptr_vo]['timestamp']
                if self.timestamp_now - pkt_timestamp > 20:
                    pkt_size = self.txq_table[uid][0][r_ptr_vo]['size']
                    self.tx_info_table[uid][0]["tot_size"] -= pkt_size
                    bytes_dropped += pkt_size
                    packets_dropped += 1
                    r_ptr_vo = (r_ptr_vo + 1) % self.queue_size
                else:
                    break
            self.tx_info_table[uid][0]["read_ptr"] = r_ptr_vo

            # VI 队列 (ac=1), 50ms
            r_ptr_vi = self.tx_info_table[uid][1]["read_ptr"]
            w_ptr_vi = self.tx_info_table[uid][1]["write_ptr"]
            while r_ptr_vi != w_ptr_vi:
                pkt_timestamp = self.txq_table[uid][1][r_ptr_vi]['timestamp']
                if self.timestamp_now - pkt_timestamp > 50:
                    pkt_size = self.txq_table[uid][1][r_ptr_vi]['size']
                    self.tx_info_table[uid][1]["tot_size"] -= pkt_size
                    bytes_dropped += pkt_size
                    packets_dropped += 1
                    r_ptr_vi = (r_ptr_vi + 1) % self.queue_size
                else:
                    break
            self.tx_info_table[uid][1]["read_ptr"] = r_ptr_vi
            
        return packets_dropped, bytes_dropped

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.txq_table = create_txq_table(self.user_num, self.queue_size)
        self.tx_info_table = create_tx_info_table(self.user_num)
        self.rate_table = create_rate_table(self.user_num) # 每局游戏使用一套新的速率
        
        self.event_queue = []
        self.timestamp_now = 0.0
        
        # 放入初始的 "traffic_generate" 事件
        heapq.heappush(self.event_queue, Task("traffic_generate", 0.0))
        
        # PPO 统计重置
        self.total_bytes_sent = 0
        self.total_packets_sent = 0
        self.total_latency = 0
        self.total_drops_full = 0
        self.total_drops_timeout = 0

        # PPO agent 在 0.5ms 时做第一次决策 (同原 schedule)
        self.timestamp_now = 0.5
        
        # 运行仿真直到第一个决策点
        while self.event_queue and self.event_queue[0].timestamp < self.timestamp_now:
            task = heapq.heappop(self.event_queue)
            # 在 PPO Env 中, traffic_generator 必须由 env 调用
            # 我们需要一个不自动重入的版本
            self._traffic_generator_step(task.timestamp)
            
        return self._get_obs(), {} # 返回 (obs, info)

    def _traffic_generator_step(self, timestamp):
        """
        这是 traffic_generator 的 PPO 环境版本
        它不操作 event_queue, 而是由 env.step() 来管理
        """
        packets_dropped_full = 0
        for uid in range(self.user_num):
            for ac_num in range(3):
                if self.rate_table[uid][ac_num][0] > 0:
                    if timestamp % self.rate_table[uid][ac_num][1] == 0:
                        if (self.tx_info_table[uid][ac_num]["write_ptr"] + 1) % self.queue_size == self.tx_info_table[uid][ac_num]["read_ptr"]:
                            packets_dropped_full += 1
                            continue
                        packet_size = np.random.poisson(lam=self.rate_table[uid][ac_num][0] * 0.001)
                        if packet_size == 0: continue
                        
                        pkt_info = {'size': packet_size, 'timestamp': timestamp}
                        
                        self.tx_info_table[uid][ac_num]["tot_size"] += packet_size
                        w_ptr = self.tx_info_table[uid][ac_num]["write_ptr"]
                        self.txq_table[uid][ac_num][w_ptr] = pkt_info
                        self.tx_info_table[uid][ac_num]["write_ptr"] = (w_ptr + 1) % self.queue_size
                        
        heapq.heappush(self.event_queue, Task("traffic_generate", timestamp + 1))
        return packets_dropped_full

    def step(self, action):
        """执行一个 PPO 动作"""
        
        decision_timestamp = self.timestamp_now
        
        # --- 1. 解析和应用 PPO 动作 ---
        
        sched_uid = -1
        sched_tid = -1
        is_no_op = False
        
        if action == self.no_op_action:
            is_no_op = True
        else:
            # action 1...N 映射到 (uid, tid)
            action_idx = action - 1
            sched_uid = action_idx // self.num_acs
            sched_tid = action_idx % self.num_acs
            
            # 检查 PPO 是否选了空队列
            if self.tx_info_table[sched_uid][sched_tid]["read_ptr"] == self.tx_info_table[sched_uid][sched_tid]["write_ptr"]:
                is_no_op = True
        
        bytes_sent = 0
        packets_sent = 0
        latency_sum = 0.0
        send_time = 0.0
        
        if is_no_op:
            send_time = 0.1 # 无操作，推进 0.1ms (惩罚)
        else:
            # --- 复用原 schedule 函数的调度执行逻辑 ---
            temp_sched_size = 0
            r_ptr = self.tx_info_table[sched_uid][sched_tid]["read_ptr"]
            w_ptr = self.tx_info_table[sched_uid][sched_tid]["write_ptr"]
            
            while r_ptr != w_ptr:
                if temp_sched_size == self.max_sched_bytes:
                    break
                
                pkt = self.txq_table[sched_uid][sched_tid][r_ptr]
                pkt_size = pkt["size"]
                
                if temp_sched_size + pkt_size <= self.max_sched_bytes:
                    temp_sched_size += pkt_size
                    latency_sum += (decision_timestamp - pkt["timestamp"])
                    packets_sent += 1
                    r_ptr = (r_ptr + 1) % self.queue_size
                else:
                    # 报文切分
                    remaining_bytes = self.max_sched_bytes - temp_sched_size
                    self.txq_table[sched_uid][sched_tid][r_ptr]["size"] = pkt_size - remaining_bytes
                    
                    # 仅记录被切分部分的时延 (按比例)
                    latency_sum += (decision_timestamp - pkt["timestamp"]) * (remaining_bytes / pkt_size)
                    # (不计为一个完整的包)
                    
                    temp_sched_size = self.max_sched_bytes
                    break
            
            bytes_sent = temp_sched_size
            self.tx_info_table[sched_uid][sched_tid]["read_ptr"] = r_ptr
            self.tx_info_table[sched_uid][sched_tid]["tot_size"] -= bytes_sent
            
            # 保证 send_time > 0
            send_time = max(bytes_sent / self.max_rate * 1000, 0.001)

        
        # --- 2. 推进仿真到下一个决策点 ---
        
        next_decision_time = decision_timestamp + send_time
        packets_dropped_full_step = 0
        
        while self.event_queue and self.event_queue[0].timestamp <= next_decision_time:
            task = heapq.heappop(self.event_queue)
            self.timestamp_now = task.timestamp
            
            if task.event == "traffic_generate":
                packets_dropped_full_step += self._traffic_generator_step(task.timestamp)
        
        self.timestamp_now = next_decision_time # 跳到下一个决策点
        
        
        # --- 3. 检查超时丢包 (在决策点执行) ---
        packets_dropped_timeout_step, bytes_dropped_timeout_step = self._discard_old_packets()
        
        
        # --- 4. 计算 PPO 奖励 (Reward) ---
        
        # 奖励 = 吞吐量 (正面)
        # 惩罚 = 时延 (负面)
        # 惩罚 = 丢包 (负面)
        
        # 吞吐量奖励 (按 KB)
        throughput_reward = bytes_sent / 1024.0
        
        # 时延惩罚 (ms)
        latency_penalty = latency_sum
        
        # 丢包惩罚 (每个包 -10)
        drop_penalty = (packets_dropped_full_step + packets_dropped_timeout_step) * 10.0
        
        # 选空队列惩罚
        no_op_penalty = 1.0 if (action != self.no_op_action and is_no_op) else 0.0
        
        reward = throughput_reward - latency_penalty - drop_penalty - no_op_penalty

        # --- 5. 更新统计 & 准备返回 ---
        
        self.total_bytes_sent += bytes_sent
        self.total_packets_sent += packets_sent
        self.total_latency += latency_sum
        self.total_drops_full += packets_dropped_full_step
        self.total_drops_timeout += packets_dropped_timeout_step
        
        done = self.timestamp_now >= self.episode_length
        
        info = {}
        if done:
            info = {
                "episode_stats": {
                    "bytes_sent_MB": self.total_bytes_sent / (1024*1024),
                    "packets_sent_K": self.total_packets_sent / 1000,
                    "avg_latency_ms": (self.total_latency / self.total_packets_sent) if self.total_packets_sent > 0 else 0,
                    "drops_full_K": self.total_drops_full / 1000,
                    "drops_timeout_K": self.total_drops_timeout / 1000
                }
            }
            
        obs = self._get_obs()
        
        # (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info


# --- 3. PPO Agent 实现 ---

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 临时存储 (Replay Buffer)
        self.buffer = []

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.buffer.append((state, action, log_prob, reward, done, value))

    def clear_buffer(self):
        self.buffer = []

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action, log_prob, _ = self.actor.get_action(state_tensor)
            value = self.critic.get_value(state_tensor)
        return action.item(), log_prob.item(), value.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for state, action, log_prob, reward, done, value in reversed(self.buffer):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.stack([torch.FloatTensor(t[0]) for t in self.buffer]).to(self.device)
        old_actions = torch.tensor([t[1] for t in self.buffer], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([t[2] for t in self.buffer], dtype=torch.float32).to(self.device)
        old_values = torch.tensor([t[5] for t in self.buffer], dtype=torch.float32).to(self.device)

        # Calculate advantages
        advantages = rewards - old_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            log_probs, state_values, dist_entropy = self.actor.get_action(old_states, old_actions)
            state_values = self.critic.get_value(old_states).squeeze()

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # (PPO-Clip loss + Value loss - Entropy bonus)
            loss = (-torch.min(surr1, surr2)
                    + 0.5 * nn.MSELoss()(state_values, rewards)
                    - 0.01 * dist_entropy)

            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.clear_buffer()

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
        print(f"PPO model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"PPO model loaded from {path}")

# ##################################################################
# --- (结束) PPO 及 Gym 环境整合代码 ---
# ##################################################################


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run simulation with Traditional or PPO scheduler.")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="traditional", 
        choices=["traditional", "ppo_train", "ppo_eval"],
        help="Mode to run: 'traditional' (original), 'ppo_train', or 'ppo_eval'."
    )
    args = parser.parse_args()

    # --- 仿真参数 ---
    user_num = 32
    queue_size = 512 #每个tid队列缓存的报文数量
    duration_per_time = 10000 # 10 秒
    
    
    # #############################################
    # --- 模式 1: 运行原始的传统调度器 ---
    # #############################################
    if args.mode == "traditional":
        print("Running in 'traditional' mode...")
        
        txq_table = create_txq_table(user_num, queue_size)
        tx_info_table = create_tx_info_table(user_num)
        
        # 事件队列
        priority_queue = []
        heapq.heappush(priority_queue, Task("traffic_generate", 0))
        heapq.heappush(priority_queue, Task("schedule", 0.5))
        
        runtimes = 100 # 运行 100 * 10000 ms
        
        rate_table = create_rate_table(user_num)
        
        last_print_time = 0
        
        # --- TQDM 和 统计 ---
        stats_dict = {
            'bytes_sent': 0,
            'packets_sent': 0,
            'latency_sum': 0,
            'drops_full': 0,
            'drops_timeout': 0,
            'drops_timeout_bytes': 0
        }
        total_duration_ms = runtimes * duration_per_time
        pbar = tqdm(total=total_duration_ms, unit="ms", desc="Traditional Sim")

        while True:
            if not priority_queue:
                print("Event queue empty, simulation ended.")
                break
                
            task = heapq.heappop(priority_queue)
            
            # 更新 TQDM 进度条
            current_time = task.timestamp
            if current_time <= total_duration_ms:
                pbar.update(current_time - pbar.n) # 更新到当前时间
            
            if math.floor(current_time) > last_print_time:
                last_print_time = math.floor(current_time)
                if last_print_time % 1000 == 0: # 每 1000ms 更新一次性能展示
                    avg_latency = (stats_dict['latency_sum'] / stats_dict['packets_sent']) if stats_dict['packets_sent'] > 0 else 0
                    throughput_mbps = (stats_dict['bytes_sent'] * 8 / (1024*1024)) / (current_time / 1000) if current_time > 0 else 0
                    pbar.set_postfix({
                        "AvgLat(ms)": f"{avg_latency:.2f}",
                        "Tput(Mbps)": f"{throughput_mbps:.2f}",
                        "Drops(Full)": f"{stats_dict['drops_full']}",
                        "Drops(Timeout)": f"{stats_dict['drops_timeout']}"
                    })

            if task.event == "traffic_generate":
                # 注意: 原始的 traffic_generator 会自动重新入队
                traffic_generator(txq_table, user_num, task.timestamp, rate_table, tx_info_table, queue_size, priority_queue, stats_dict)
            else:
                # 原始的 schedule 会自动重新入队
                schedule(txq_table, tx_info_table, user_num, task.timestamp, queue_size, priority_queue, stats_dict)
           
            if math.floor(task.timestamp) % duration_per_time == 0 and math.floor(task.timestamp) > 0:
                tqdm.write(f"Timestamp {task.timestamp:.2f}: Regenerating rate table...")
                rate_table = create_rate_table(user_num)
                # 重置统计
                stats_dict = {k: 0 for k in stats_dict}

            if task.timestamp > total_duration_ms:
                pbar.close()
                tqdm.write(f"Simulation finished at {task.timestamp:.2f} ms.")
                break
    
    
    # #############################################
    # --- 模式 2 & 3: 运行 PPO 训练或评估 ---
    # #############################################
    elif args.mode in ["ppo_train", "ppo_eval"]:
        
        # --- PPO 超参数 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running PPO on device: {device}")
        
        lr_actor = 0.0003
        lr_critic = 0.001
        gamma = 0.99       # 折扣因子
        K_epochs = 80      # 每次 update 迭代次数
        eps_clip = 0.2     # PPO clip 范围
        update_timestep = 4096 # 每 N 步更新一次网络
        
        total_episodes = 500 # 总训练局数
        episode_length_ms = duration_per_time # 每局长度 (10000 ms)
        
        model_save_path = "./ppo_scheduler.pth"

        # --- 初始化环境和 Agent ---
        env = SimulationEnv(
            user_num=user_num, 
            queue_size=queue_size, 
            episode_length=episode_length_ms
        )
        
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            K_epochs=K_epochs,
            eps_clip=eps_clip,
            device=device
        )

        if args.mode == "ppo_eval":
            print("Running in 'ppo_eval' mode...")
            try:
                agent.load_model(model_save_path)
                agent.actor.eval()
                agent.critic.eval()
            except FileNotFoundError:
                print(f"Error: Model file not found at {model_save_path}. Please run 'ppo_train' first.")
                exit(1)
            
            total_episodes = 10 

        else:
            print("Running in 'ppo_train' mode...")
        

        # --- 训练/评估循环 ---
        time_step = 0
        
        outer_pbar = tqdm(range(total_episodes), desc=f"Running {args.mode}") 
        
        for episode in outer_pbar:
            state, info = env.reset()
            current_ep_reward = 0
            
            # TQDM 内循环
            inner_pbar = tqdm(total=episode_length_ms, desc=f"Episode {episode}", unit="ms", leave=False) 
            
            done = False
            status_message = "" # 用于在进度条上显示状态
            while not done:
                # --- Agent 决策 ---
                action, log_prob, value = agent.choose_action(state)

                # --- 环境执行 ---
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                current_ep_reward += reward
                
                # 准备要显示在进度条上的性能数据
                postfix_data = {"Reward": f"{current_ep_reward:.1f}"}
                
                if args.mode == "ppo_train":
                    # 存储经验
                    agent.store_transition(state, action, log_prob, reward, done, value)
                    time_step += 1

                    # 如果 buffer 满了, 更新网络
                    if time_step % update_timestep == 0:
                        status_message = "Updating..." # 1. 设置状态
                        postfix_data["Status"] = status_message
                        inner_pbar.set_postfix(postfix_data) # 2. 立即更新进度条以显示 "Updating..."                        
                        
                        # 计算最后一个状态的 Value
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
                            last_value = agent.critic.get_value(state_tensor).item()
                        agent.update() # 3. 执行耗时的更新
                        status_message = "" # 4. 清除状态
                    
                    # 更新状态 (如果 status_message 为空, 就移除 Status 键)
                    if status_message:
                        postfix_data["Status"] = status_message
                    else:
                        postfix_data.pop("Status", None)
                
                # 在循环的最后，统一更新进度条和时间
                inner_pbar.set_postfix(postfix_data)
                inner_pbar.update(env.timestamp_now - inner_pbar.n)
                
            # --- 一局结束, 关闭内循环 pbar, 打印统计信息 ---
            inner_pbar.close()
            
            if "episode_stats" in info:
                stats = info['episode_stats']

                tqdm.write(f"--- Episode {episode} Finished ({args.mode}) ---")
                tqdm.write(f"  Total Reward: {current_ep_reward:.2f}")
                tqdm.write(f"  Avg Latency: {stats['avg_latency_ms']:.2f} ms")
                tqdm.write(f"  Throughput: {stats['bytes_sent_MB']:.2f} MB")
                tqdm.write(f"  Drops (Full): {stats['drops_full_K']:.1f} K packets")
                tqdm.write(f"  Drops (Timeout): {stats['drops_timeout_K']:.1f} K packets")

                outer_pbar.set_postfix({
                    "AvgLat": f"{stats['avg_latency_ms']:.1f}ms",
                    "Tput": f"{stats['bytes_sent_MB']:.1f}MB",
                    "Drops": f"{stats['drops_full_K']+stats['drops_timeout_K']:.1f}K"
                })
            
            if args.mode == "ppo_train" and (episode + 1) % 50 == 0:
                # 每 50 局保存一次模型
                agent.save_model(model_save_path)

        outer_pbar.close()
        env.close()
        print(f"Finished {args.mode}.")