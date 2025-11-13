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

# ==== 新增：RL & 可视化相关 ====
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
class Task:
    def __init__(self, event, timestamp):
        self.event = event
        self.timestamp = timestamp
    def __lt__(self, other):
        # 定义小于号的比较方式，用于 heapq 排序
        return self.timestamp < other.timestamp
    def __repr__(self):
        return f"Task(name={self.name}, priority={self.priority})"
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
def gen_sched_res(txq_table, tx_info_table, user_num, timestamp_now, queue_size):
    vo_list = []
    vi_list = []
    be_list = []
    #首先从txq_table中取首包做排序
    schedule_res = {"uid" : -1, "ac_type" : -1}
    for uid in range(user_num):
        for ac in range(3):
            w_ptr = tx_info_table[uid][ac]["write_ptr"]
            r_ptr = tx_info_table[uid][ac]["read_ptr"]
            if w_ptr != r_ptr: #队列非空
                pkt_info = txq_table[uid][ac][r_ptr]
                #txq_table[uid][ac].pop(0)
                schedule_info = {
                    "uid" : uid,
                    "ac_type" : ac,
                    "size" : tx_info_table[uid][ac]["tot_size"],
                    "timestamp" : pkt_info['timestamp']
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
    #尝试调度VO
    #print(f"timestamp {timestamp_now} vo queue length is {len(sorted_vo_list)}")
    for i in range(len(sorted_vo_list)):
        if timestamp_now - sorted_vo_list[i]["timestamp"] <= 20 and sorted_vo_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_vo_list[i]['uid'])}, ac type is vo")
            schedule_res = {"uid" : sorted_vo_list[i]['uid'], "ac_type" : 0}
            return schedule_res
        elif timestamp_now - sorted_vo_list[i]["timestamp"] > 20:
            #丢弃超时报文
            uid = sorted_vo_list[i]['uid']
            r_ptr = tx_info_table[uid][0]["read_ptr"]
            w_ptr = tx_info_table[uid][0]["write_ptr"]
            while r_ptr != w_ptr:
                if timestamp_now - txq_table[uid][0][r_ptr]['timestamp'] > 20:
                    #记录丢弃长度
                    tx_info_table[uid][0]["tot_size"] = tx_info_table[uid][0]["tot_size"] - txq_table[uid][0][r_ptr]['size']
                    r_ptr = (r_ptr + 1) % queue_size
                else:
                    break
            tx_info_table[uid][0]["read_ptr"] = r_ptr
    #尝试调度VI
    for i in range(len(sorted_vi_list)):
        if timestamp_now - sorted_vi_list[i]["timestamp"] > 20 and timestamp_now - sorted_vi_list[i]["timestamp"] <= 50 and sorted_vi_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_vo_list[i]['uid'])}, ac type is vi")
            schedule_res = {"uid" : sorted_vi_list[i]['uid'], "ac_type" : 1}
            return schedule_res
        elif timestamp_now - sorted_vi_list[i]["timestamp"] > 50:
            #丢弃超时报文
            uid = sorted_vo_list[i]['uid']
            r_ptr = tx_info_table[uid][1]["read_ptr"]
            w_ptr = tx_info_table[uid][1]["write_ptr"]
            while r_ptr != w_ptr:
                if timestamp_now - txq_table[uid][1][r_ptr]['timestamp'] > 50:
                    #记录丢弃长度
                    tx_info_table[uid][1]["tot_size"] = tx_info_table[uid][1]["tot_size"] - txq_table[uid][1][r_ptr]['size']
                    r_ptr = (r_ptr + 1) % queue_size
                else:
                    break
            tx_info_table[uid][1]["read_ptr"] = r_ptr
    #调度BE，假定所有用户的最小需求都是100ms调度一次
    for i in range(len(sorted_be_list)):
        if timestamp_now - sorted_be_list[i]["timestamp"] > 100 and sorted_be_list[i]["timestamp"] < timestamp_now:
            #print(f"schedule :: uid is {(sorted_be_list[i]['uid'])}, ac type is be")
            schedule_res = {"uid" : sorted_be_list[i]['uid'], "ac_type" : 2}
            return schedule_res
    #没有调度到BE，调度剩余报文最长的用户
    max_pkt_size = 0
    res_uid = -1
    res_ac_type = -1
    #这个时候队列可能已经发生变化
    for i in range(user_num):
        if tx_info_table[i][1]["tot_size"] > max_pkt_size:
            res_uid = i
            max_pkt_size = tx_info_table[i][1]["tot_size"]
            res_ac_type = 1
    for i in range(user_num):
        if tx_info_table[i][2]["tot_size"] > max_pkt_size:
            res_uid = i
            max_pkt_size = tx_info_table[i][2]["tot_size"]
            res_ac_type = 2
    #print(f"schedule :: uid is {res_uid}, ac type is {res_ac_type}, pkt_size = {max_pkt_size}")
    #print(f"get_schres:: r_ptr is {tx_info_table[res_uid][res_ac_type]['read_ptr']}, w_ptr is {tx_info_table[res_uid][res_ac_type]['write_ptr']}, remain_size is {tx_info_table[res_uid][res_ac_type]['tot_size']}")
    schedule_res = {"uid" : res_uid, "ac_type" : res_ac_type}
    return schedule_res
   
def schedule(txq_table, tx_info_table, user_num, timestamp, queue_size, event_queue):
    max_rate = 2442 * 1000 * 1000
    #先尝试调度VO
    #all_empty = True
    #print("call gen_sched_res")
    sched_res = gen_sched_res(txq_table, tx_info_table, user_num, timestamp, queue_size)
    '''
    globals_dict = {
        'txq_table': txq_table,
        'tx_info_table': tx_info_table,
        'user_num': user_num,
        'start_time': start_time,
        'queue_size': queue_size,
        'gen_sched_res': gen_sched_res # 确保函数也在子进程中可用
    }
    # 使用 runctx 代替 run，并传入 globals 和locals
    cProfile.runctx(
        'gen_sched_res(txq_table, tx_info_table, user_num, start_time, queue_size)',
        globals_dict,
        {},
        sort='time'
    )
    '''
    #cProfile.run('gen_sched_res(txq_table, tx_info_table, user_num, start_time, queue_size)')
    sched_uid = sched_res["uid"]
    sched_tid = sched_res["ac_type"]
    if sched_uid == -1:
        print(f"nothing to schedule!!")
        heapq.heappush(event_queue, Task("schedule", timestamp + 1))
        return
    print(f"schedule uid is {sched_uid}, tid is {sched_tid}")
    temp_sched_size = 0
    r_ptr = tx_info_table[sched_uid][sched_tid]["read_ptr"]
    w_ptr = tx_info_table[sched_uid][sched_tid]["write_ptr"]
    #print(f"r_ptr is {r_ptr}, w_ptr is {w_ptr}")
    while r_ptr != w_ptr:
        if temp_sched_size == max_rate / 1000 * 4:
            break
        if temp_sched_size + txq_table[sched_uid][sched_tid][r_ptr]["size"] <= max_rate / 1000 * 4:
            temp_sched_size = temp_sched_size + txq_table[sched_uid][sched_tid][r_ptr]["size"]
            #print(f"r_ptr is {r_ptr}, packet_size is {txq_table[sched_uid][sched_tid][0]['size']}, sched_size is {temp_sched_size}")
            r_ptr = (r_ptr + 1) % queue_size
        else:
            txq_table[sched_uid][sched_tid][r_ptr]["size"] = txq_table[sched_uid][sched_tid][r_ptr]["size"] - max_rate / 1000 * 4 + temp_sched_size
            temp_sched_size = max_rate / 1000 * 4
    tx_info_table[sched_uid][sched_tid]["read_ptr"] = r_ptr
   
    tx_info_table[sched_uid][sched_tid]["tot_size"] = tx_info_table[sched_uid][sched_tid]["tot_size"] - temp_sched_size
    send_time = temp_sched_size / max_rate * 1000
    if send_time == 0:
        import pdb;pdb.set_trace()
    #print(f"schedule complete timestamp is {timestamp}, send_time is {send_time}, temp_sched_size is {temp_sched_size}, queue remain size is {tx_info_table[sched_uid][sched_tid]['tot_size']}")
   
    heapq.heappush(event_queue, Task("schedule", timestamp + send_time))
    #print(f"sleep time is {send_time / 1000}")
def traffic_generator(txq_table, user_num, timestamp, rate_table, tx_info_table, queue_size, event_queue):
    for uid in range(user_num):
        for ac_num in range(3):
            if rate_table[uid][ac_num][0] > 0:
                #速率表中指定了发包间隔
                if timestamp % rate_table[uid][ac_num][1] == 0:
                    if (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size == tx_info_table[uid][ac_num]["read_ptr"]:
                        print(f"time {timestamp} uid {uid}, tid {ac_num} queue is full, drop input packet")
                        continue
                    packet_size = np.random.poisson(lam=rate_table[uid][ac_num][0] * 0.001)
                    pkt_info = {}
                    pkt_info['size'] = packet_size
                    pkt_info['timestamp'] = timestamp
                    #存储当前队列缓存包长度
                    #print(f"traffic generate uid : {uid}, ac_type : {ac_num}, pkt_size : {packet_size}, queue_remain len : {tx_info_table[uid][ac_num]['tot_size']}")
                    tx_info_table[uid][ac_num]["tot_size"] = tx_info_table[uid][ac_num]["tot_size"] + packet_size
                    w_ptr = tx_info_table[uid][ac_num]["write_ptr"]
                    txq_table[uid][ac_num][w_ptr] = pkt_info
                    tx_info_table[uid][ac_num]["write_ptr"] = (tx_info_table[uid][ac_num]["write_ptr"] + 1) % queue_size
    heapq.heappush(event_queue, Task("traffic_generate", timestamp + 1))
#def create_stat_table(user_num):
       
# ==============================
#      PPO 算法与调度环境
# ==============================

@dataclass
class PPOConfig:
    user_num: int = 32
    queue_size: int = 2048            # 加大队列长度，减小溢出的概率
    max_sim_time: float = 1e5         # 每个 episode 的最大仿真时间（毫秒）
    gamma: float = 0.99
    lam: float = 0.95                 # GAE lambda
    clip_eps: float = 0.2
    lr: float = 3e-4
    epochs: int = 10                  # 每个 episode 的 PPO 更新轮数
    rollout_steps: int = 1024         # 每轮收集的步数
    minibatch_size: int = 256
    total_episodes: int = 200
    w_vo: float = 3.0
    w_vi: float = 2.0
    w_be: float = 1.0
    seed: int = 42
    num_envs: int = 1                 # 向量环境个数，>1 时可以扩展为多进程
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seeds(seed: int):
    """统一设置 random / numpy / torch 的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RLSchedulingEnv:
    """
    强化学习版本的调度环境：
    - 复用原有 create_txq_table / create_tx_info_table / create_rate_table / traffic_generator
    - 调度时可以选择：
      * use_traditional=True：使用原 gen_sched_res 规则
      * use_traditional=False：使用 RL 动作 (uid, tid)
    """

    def __init__(self, config: PPOConfig, use_traditional: bool = False):
        self.cfg = config
        self.user_num = config.user_num
        self.queue_size = config.queue_size
        self.max_sim_time = config.max_sim_time
        self.use_traditional = use_traditional

        self.txq_table = None
        self.tx_info_table = None
        self.rate_table = None
        self.event_queue = None
        self.current_time = 0.0

        # obs 维度：每个队列两个特征 + 当前时间
        self.obs_dim = self.user_num * 3 * 2 + 1
        self.act_dim = self.user_num * 3

    def reset(self, seed: int = None):
        """环境重置，初始化队列和事件"""
        if seed is not None:
            set_global_seeds(seed)
        # 初始化队列和统计信息
        self.txq_table = create_txq_table(self.user_num, self.queue_size)
        self.tx_info_table = create_tx_info_table(self.user_num)
        self.rate_table = create_rate_table(self.user_num)
        self.event_queue = []
        heapq.heappush(self.event_queue, Task("traffic_generate", 0.0))
        heapq.heappush(self.event_queue, Task("schedule", 0.5))
        self.current_time = 0.0
        self.done = False
        return self._build_observation()

    def _build_observation(self):
        """构造当前状态向量：每个 (uid, tid) 的 tot_size 和头部时延 + 当前时间"""
        features = []
        for uid in range(self.user_num):
            for tid in range(3):
                info = self.tx_info_table[uid][tid]
                tot_size = info["tot_size"]
                r_ptr = info["read_ptr"]
                w_ptr = info["write_ptr"]
                if r_ptr != w_ptr:
                    head_ts = self.txq_table[uid][tid][r_ptr]['timestamp']
                    delay = max(0.0, self.current_time - head_ts)
                else:
                    delay = 0.0
                features.append(float(tot_size))
                features.append(float(delay))
        # 加上当前时间（也可以归一化）
        features.append(float(self.current_time))
        return np.array(features, dtype=np.float32)

    def _compute_max_delays(self):
        """计算三类业务的最大头部时延"""
        max_vo = 0.0
        max_vi = 0.0
        max_be = 0.0
        for uid in range(self.user_num):
            for tid in range(3):
                info = self.tx_info_table[uid][tid]
                r_ptr = info["read_ptr"]
                w_ptr = info["write_ptr"]
                if r_ptr != w_ptr:
                    head_ts = self.txq_table[uid][tid][r_ptr]['timestamp']
                    d = max(0.0, self.current_time - head_ts)
                    if tid == 0:
                        max_vo = max(max_vo, d)
                    elif tid == 1:
                        max_vi = max(max_vi, d)
                    else:
                        max_be = max(max_be, d)
        return max_vo, max_vi, max_be

    def step(self, action: int = 0):
        """
        执行一步：
        1. 从事件队列中推进到下一个 schedule 事件，期间处理所有 traffic_generate
        2. 在 schedule 时刻按传统 or RL 规则进行一次调度
        3. 返回新的状态、reward、done、info
        """
        if self.done:
            raise RuntimeError("Env is done, call reset() first")

        # 1. 先把时间推进到下一个调度时刻
        while True:
            if not self.event_queue:
                self.done = True
                return self._build_observation(), 0.0, True, {}
            task = heapq.heappop(self.event_queue)
            self.current_time = task.timestamp

            # 动态调整速率（保留原逻辑：每 duration_per_time 新生一套 rate_table）
            duration_per_time = 10000
            if math.floor(self.current_time) % duration_per_time == 0:
                self.rate_table = create_rate_table(self.user_num)

            if task.event == "traffic_generate":
                traffic_generator(
                    self.txq_table,
                    self.user_num,
                    task.timestamp,
                    self.rate_table,
                    self.tx_info_table,
                    self.queue_size,
                    self.event_queue,
                )
            elif task.event == "schedule":
                # 到调度时刻，跳出
                break

        # 2. 执行调度，决定 sched_uid / sched_tid
        if self.use_traditional:
            # 传统调度规则：直接用 gen_sched_res
            sched_res = gen_sched_res(
                self.txq_table,
                self.tx_info_table,
                self.user_num,
                self.current_time,
                self.queue_size
            )
            sched_uid = sched_res["uid"]
            sched_tid = sched_res["ac_type"]
        else:
            # RL 动作映射到 (uid, tid)
            idx = int(action)
            if idx < 0 or idx >= self.act_dim:
                idx = 0
            sched_uid = idx // 3
            sched_tid = idx % 3

            # 如果该 (uid, tid) 队列为空，为了安全回退到传统调度
            info = self.tx_info_table[sched_uid][sched_tid]
            if info["read_ptr"] == info["write_ptr"]:
                sched_res = gen_sched_res(
                    self.txq_table,
                    self.tx_info_table,
                    self.user_num,
                    self.current_time,
                    self.queue_size
                )
                sched_uid = sched_res["uid"]
                sched_tid = sched_res["ac_type"]

        # 3. 发包逻辑：完全复用原 schedule() 的实现，只是把选择队列部分换成上面的 sched_uid, sched_tid
        max_rate = 2442 * 1000 * 1000
        if sched_uid == -1:
            # 无包可发
            heapq.heappush(self.event_queue, Task("schedule", self.current_time + 1.0))
            # reward 直接基于当前队列状态
            max_vo, max_vi, max_be = self._compute_max_delays()
            delay_cost = (self.cfg.w_vo * max_vo +
                          self.cfg.w_vi * max_vi +
                          self.cfg.w_be * max_be)
            # 适当裁剪和缩放，提升收敛性
            delay_cost = min(delay_cost, 200.0)
            reward = -delay_cost / 100.0
            done = self.current_time > self.max_sim_time
            self.done = done
            return self._build_observation(), reward, done, {
                "sched_uid": sched_uid,
                "sched_tid": sched_tid,
                "time": self.current_time,
            }

        temp_sched_size = 0.0
        r_ptr = self.tx_info_table[sched_uid][sched_tid]["read_ptr"]
        w_ptr = self.tx_info_table[sched_uid][sched_tid]["write_ptr"]
        while r_ptr != w_ptr:
            if temp_sched_size == max_rate / 1000 * 4:
                break
            pkt_size = self.txq_table[sched_uid][sched_tid][r_ptr]["size"]
            if temp_sched_size + pkt_size <= max_rate / 1000 * 4:
                temp_sched_size += pkt_size
                r_ptr = (r_ptr + 1) % self.queue_size
            else:
                # 部分发送
                self.txq_table[sched_uid][sched_tid][r_ptr]["size"] = \
                    pkt_size - (max_rate / 1000 * 4 - temp_sched_size)
                temp_sched_size = max_rate / 1000 * 4

        self.tx_info_table[sched_uid][sched_tid]["read_ptr"] = r_ptr
        self.tx_info_table[sched_uid][sched_tid]["tot_size"] -= temp_sched_size

        send_time = temp_sched_size / max_rate * 1000.0
        if send_time <= 0:
            send_time = 0.1  # 防止除零

        heapq.heappush(self.event_queue, Task("schedule", self.current_time + send_time))

        # 4. 计算 reward
        max_vo, max_vi, max_be = self._compute_max_delays()
        delay_cost = (self.cfg.w_vo * max_vo +
                      self.cfg.w_vi * max_vi +
                      self.cfg.w_be * max_be)
        delay_cost = min(delay_cost, 200.0)
        reward = -delay_cost / 100.0

        # 5. 终止判断
        done = self.current_time > self.max_sim_time
        self.done = done

        obs = self._build_observation()
        info = {
            "sched_uid": sched_uid,
            "sched_tid": sched_tid,
            "time": self.current_time,
            "max_vo_delay": max_vo,
            "max_vi_delay": max_vi,
            "max_be_delay": max_be,
        }
        return obs, reward, done, info


class ActorCritic(nn.Module):
    """简单 MLP 的 Actor-Critic 网络"""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 256
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def ppo_collect_rollout(env: RLSchedulingEnv,
                        model: nn.Module,
                        cfg: PPOConfig,
                        device: torch.device):
    """
    收集一批 rollout 数据（单环境版本，方便理解；需要多核时可扩展为向量环境）
    """
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    for _ in range(cfg.rollout_steps):
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        next_obs, reward, done, info = env.step(action.item())

        obs_buf.append(obs)
        act_buf.append(action.cpu().numpy())
        logp_buf.append(logp.cpu().numpy())
        rew_buf.append(reward)
        val_buf.append(value.cpu().numpy())
        done_buf.append(done)

        obs = next_obs
        if done:
            obs = env.reset()

    # 转成 tensor
    obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(act_buf).squeeze(-1), dtype=torch.long, device=device)
    logp_t = torch.tensor(np.array(logp_buf).squeeze(-1), dtype=torch.float32, device=device)
    rew_t = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device)
    val_t = torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)
    done_t = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)

    # GAE-Lambda 计算 advantage
    adv_buf = torch.zeros_like(rew_t, device=device)
    ret_buf = torch.zeros_like(rew_t, device=device)
    last_adv = 0.0
    last_ret = 0.0
    for t in reversed(range(cfg.rollout_steps)):
        mask = 1.0 - done_t[t]
        delta = rew_t[t] + cfg.gamma * (val_t[t + 1] if t + 1 < cfg.rollout_steps else 0.0) * mask - val_t[t]
        last_adv = delta + cfg.gamma * cfg.lam * mask * last_adv
        adv_buf[t] = last_adv
        last_ret = val_t[t] + last_adv
        ret_buf[t] = last_ret

    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

    return obs_t, act_t, logp_t, adv_buf, ret_buf


def ppo_update(model: nn.Module,
               optimizer: optim.Optimizer,
               cfg: PPOConfig,
               batch_obs, batch_act, batch_logp_old, batch_adv, batch_ret,
               device: torch.device):
    N = batch_obs.size(0)
    idxs = np.arange(N)
    for _ in range(cfg.epochs):
        np.random.shuffle(idxs)
        for start in range(0, N, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            mb_idx = idxs[start:end]

            obs = batch_obs[mb_idx]
            act = batch_act[mb_idx]
            logp_old = batch_logp_old[mb_idx]
            adv = batch_adv[mb_idx]
            ret = batch_ret[mb_idx]

            logits, value = model(obs)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (ret - value).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def train_ppo(cfg: PPOConfig):
    """主训练入口：训练 PPO 并画出奖励收敛图"""
    device = torch.device(cfg.device)
    set_global_seeds(cfg.seed)

    # 环境 & 模型
    env = RLSchedulingEnv(cfg, use_traditional=False)
    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    episode_rewards = []

    # 使用 tqdm 展示 episode 进度
    for ep in tqdm(range(cfg.total_episodes), desc="PPO Training"):
        # 收集一批 rollout
        obs_t, act_t, logp_t, adv_t, ret_t = ppo_collect_rollout(env, model, cfg, device)

        # 估计该批次的平均 reward（粗略当作一个 episode 的回报统计）
        with torch.no_grad():
            avg_ep_rew = ret_t.mean().item()
        episode_rewards.append(avg_ep_rew)

        # PPO 更新
        ppo_update(model, optimizer, cfg, obs_t, act_t, logp_t, adv_t, ret_t, device)

    # 画奖励收敛图
    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(episode_rewards, label="Episode mean return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.legend()
    fig_path = os.path.join("results", "ppo_reward_curve.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[PPO] 收敛曲线已保存到 {fig_path}")

    return model, episode_rewards


def evaluate_policy(env_cfg: PPOConfig, model: nn.Module = None, use_traditional: bool = False, seed: int = 1234):
    """
    对策略进行评估：
    - use_traditional=True：传统调度
    - use_traditional=False 且提供 model：用 PPO 策略
    """
    device = torch.device(env_cfg.device)
    set_global_seeds(seed)
    env = RLSchedulingEnv(env_cfg, use_traditional=use_traditional)
    obs = env.reset(seed=seed)
    total_reward = 0.0

    while True:
        if use_traditional:
            # 动作无意义，内部会走 gen_sched_res
            action = 0
        else:
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_t)
                # 评估时用贪心动作，保证确定性
                action = torch.argmax(logits, dim=-1).item()

        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


if __name__ == "__main__":
    """
    运行方式示例（命令行）：
    1. 仅跑原始传统调度仿真（不训练）：
       python simcore.py --mode baseline

    2. 训练 PPO：
       python simcore.py --mode train_ppo

    3. 训练后对比传统调度与 PPO（使用相同流量随机序列）：
       python simcore.py --mode compare
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=["baseline", "train_ppo", "compare"],
                        default="baseline")
    args = parser.parse_args()

    if args.mode == "baseline":
        # 原始传统调度仿真（保持你的原逻辑，只是打包成函数）
        user_num = 32
        simu_time = 10000  # 毫秒为单位
        queue_size = 512   # 使用原来的队列大小
        txq_table = create_txq_table(user_num, queue_size)
        tx_info_table = create_tx_info_table(user_num)

        priority_queue = []
        heapq.heappush(priority_queue, Task("traffic_generate", 0.0))
        heapq.heappush(priority_queue, Task("schedule", 0.5))
        duration_per_time = 10000
        runtimes = 100

        rate_table = create_rate_table(user_num)
        while True:
            task = heapq.heappop(priority_queue)
            if task.event == "traffic_generate":
                traffic_generator(
                    txq_table, user_num, task.timestamp,
                    rate_table, tx_info_table, queue_size,
                    priority_queue
                )
            else:
                schedule(
                    txq_table, tx_info_table, user_num,
                    task.timestamp, queue_size, priority_queue
                )

            if math.floor(task.timestamp) % duration_per_time == 0:
                rate_table = create_rate_table(user_num)
            if task.timestamp > runtimes * duration_per_time:
                break

    elif args.mode == "train_ppo":
        cfg = PPOConfig()
        model, rewards = train_ppo(cfg)

    elif args.mode == "compare":
        # 对比传统调度和 PPO 策略，保证使用相同流量随机性
        cfg = PPOConfig()
        # 先训练一个模型（也可以改成直接 load 已训练好的参数）
        model, rewards = train_ppo(cfg)

        seed = 2024
        base_ret = evaluate_policy(cfg, model=None, use_traditional=True, seed=seed)
        ppo_ret = evaluate_policy(cfg, model=model, use_traditional=False, seed=seed)

        print(f"传统调度 总回报: {base_ret:.4f}")
        print(f"PPO 调度 总回报: {ppo_ret:.4f}")
