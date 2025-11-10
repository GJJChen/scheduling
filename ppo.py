import random
from functools import cmp_to_key
import numpy as np
import heapq
import math
import time
from typing import List, Dict, Tuple, Any
from collections import deque

# --- 深度学习与环境库 ---
# 你需要安装: pip install torch gymnasium
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

# --- 1. 原始仿真代码（稍作修改以用于环境中） ---

class Task:
    def __init__(self, event, timestamp):
        self.event = event
        self.timestamp = timestamp
    def __lt__(self, other):
        return self.timestamp < other.timestamp

def compare(a, b):
    if a["rate"] < b["rate"]: return 1
    return -1

def cmp(a, b):
    if a["timestamp"] > b["timestamp"]: return 1
    elif a["timestamp"] == b["timestamp"]:
        if a["size"] < b["size"]: return 1
        else: return -1
    else: return -1

def create_txq_table(user_num, queue_size):
    txq_table = []
    for _ in range(user_num):
        tid_table = []
        for _ in range(3):
            tid_table.append([0 for _ in range(queue_size)])
        txq_table.append(tid_table)
    return txq_table

def create_tx_info_table(user_num):
    tx_info_table = []
    for _ in range(user_num):
        tx_info_tid_table = []
        for _ in range(3):
            tx_info_tid_table.append({"tot_size": 0, "read_ptr": 0, "write_ptr": 0})
        tx_info_table.append(tx_info_tid_table)
    return tx_info_table

def create_rate_table(user_num):
    rate_table = []
    max_band_width = 2442
    ofo_list = []
    for user_id in range(user_num):
        tid_rate_table = []
        for tid in range(3):
            if tid == 0: rand_rate = random.randint(1, 20)
            elif tid == 1: rand_rate = random.randint(1, 40)
            elif tid == 2: rand_rate = random.randint(1, 100)
            
            if tid == 0: rand_interval = random.randint(1, 20)
            elif tid == 1: rand_interval = random.randint(1, 10)
            else: rand_interval = random.randint(1, 3)
            
            rate_info = {'uid': user_id, 'tid': tid, 'rate': rand_rate}
            tid_rate_table.append([0, rand_interval])
            ofo_list.append(rate_info)
        rate_table.append(tid_rate_table)
    
    sorted_list = sorted(ofo_list, key=cmp_to_key(compare))
    
    sum_rate = 0
    for i in range(len(sorted_list)):
        if sum_rate + sorted_list[i]['rate'] < max_band_width:
            uid = sorted_list[i]['uid']
            tid = sorted_list[i]['tid']
            rate_table[uid][tid][0] = sorted_list[i]['rate'] * 1000 * 1000
            sum_rate = sum_rate + sorted_list[i]['rate']
            
    return rate_table

# --- 2. 性能统计 (用于最终评估) ---
statistics = {}

def reset_statistics():
    global statistics
    statistics = {
        'total_bytes_sent': [0, 0, 0], 'total_packets_sent': [0, 0, 0],
        'total_latency': [0, 0, 0], 'total_bytes_dropped': [0, 0, 0],
        'total_packets_dropped': [0, 0, 0]
    }

def print_statistics(sim_type, duration):
    print(f"\n--- 仿真性能统计: [{sim_type}] ---")
    print(f"--- 仿真总时长: {duration} ms ---")
    
    total_sent = sum(statistics['total_bytes_sent'])
    total_dropped = sum(statistics['total_bytes_dropped'])
    total_throughput_mbps = (total_sent * 8) / (duration * 1000)
    
    print(f"总吞吐量: {total_sent} 字节 ({total_throughput_mbps:.2f} Mbps)")
    print(f"总丢弃量: {total_dropped} 字节")
    
    for ac in range(3):
        ac_name = ["VO", "VI", "BE"][ac]
        packets = statistics['total_packets_sent'][ac]
        latency = statistics['total_latency'][ac]
        avg_latency = (latency / packets) if packets > 0 else 0
        
        print(f"  AC ({ac_name}):")
        print(f"    - 发送字节: {statistics['total_bytes_sent'][ac]} 字节")
        print(f"    - 发送包数: {packets} 个")
        print(f"    - 丢弃包数: {statistics['total_packets_dropped'][ac]} 个")
        print(f"    - 平均时延: {avg_latency:.4f} ms")


# --- 3. Gymnasium 仿真环境 ---

class SchedulerEnv(gym.Env):
    """
    将事件驱动的仿真器封装为 Gymnasium 环境
    """
    metadata = {"render_modes": []}

    def __init__(self, user_num, queue_size, sim_duration_ms, step_duration_ms=10):
        super().__init__()
        
        self.user_num = user_num
        self.queue_size = queue_size
        self.max_band_width_mbps = 2442
        self.max_rate_bps = self.max_band_width_mbps * 1000 * 1000
        self.bytes_per_ms = self.max_rate_bps / 1000.0

        self.total_sim_duration = sim_duration_ms # 整个 episode 的时长
        self.step_duration = step_duration_ms # 每个 PPO step 模拟多长时间
        
        # 动作空间: 选择一个 (uid, ac) 对
        # 额外+1个动作表示“不调度” (No-Op)
        self.n_actions = self.user_num * 3 + 1
        self.action_space = spaces.Discrete(self.n_actions)
        
        # 状态空间: [每个队列的tot_size, 每个队列的head_packet_age]
        # (user_num * 3 * 2) 个特征
        self.obs_dim = self.user_num * 3 * 2
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # 仿真状态变量
        self.txq_table = None
        self.tx_info_table = None
        self.rate_table = None
        self.event_queue = []
        self.current_time = 0.0

        # 统计变量 (用于计算奖励)
        self.last_step_stats = {}

    def _get_obs(self):
        """从仿真状态提取观测向量"""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0
        for uid in range(self.user_num):
            for ac in range(3):
                info = self.tx_info_table[uid][ac]
                queue_size_bytes = info["tot_size"]
                
                head_of_line_age = 0
                if info["read_ptr"] != info["write_ptr"]:
                    head_pkt_timestamp = self.txq_table[uid][ac][info["read_ptr"]]["timestamp"]
                    head_of_line_age = self.current_time - head_pkt_timestamp
                
                obs[idx] = queue_size_bytes
                obs[idx + 1] = head_of_line_age
                idx += 2
        
        # 归一化 (可选，但通常有益)
        obs[::2] /= (self.bytes_per_ms * 100) # 假设队列最大100ms的缓存
        obs[1::2] /= 1000.0 # 假设最大包时延1000ms
        return obs

    def _get_current_stats(self):
        """获取用于计算奖励的当前统计数据"""
        return {
            "bytes_sent": sum(statistics["total_bytes_sent"]),
            "bytes_dropped": sum(statistics["total_bytes_dropped"]),
            "total_latency": sum(statistics["total_latency"]),
            "packets_sent": sum(statistics["total_packets_sent"]),
        }

    def _calculate_reward(self):
        """根据两步之间的统计数据变化计算奖励"""
        new_stats = self._get_current_stats()
        
        # 计算差值
        delta_bytes_sent = new_stats["bytes_sent"] - self.last_step_stats["bytes_sent"]
        delta_bytes_dropped = new_stats["bytes_dropped"] - self.last_step_stats["bytes_dropped"]
        delta_latency = new_stats["total_latency"] - self.last_step_stats["total_latency"]
        delta_packets_sent = new_stats["packets_sent"] - self.last_step_stats["packets_sent"]
        
        # 奖励吞吐量
        # (将字节转换为“兆字节”级别，避免数值过大)
        reward_throughput = delta_bytes_sent / 1000000.0
        
        # 惩罚丢包
        penalty_dropped = delta_bytes_dropped / 1000000.0
        
        # 惩罚时延
        avg_latency_this_step = (delta_latency / delta_packets_sent) if delta_packets_sent > 0 else 0
        # (将毫秒时延缩放)
        penalty_latency = avg_latency_this_step / 100.0 
        
        # **（关键）奖励函数**
        # 这是你需要重点调优的地方
        reward = (reward_throughput * 1.0) - (penalty_dropped * 2.0) - (penalty_latency * 0.5)
        
        # 更新上一步的统计
        self.last_step_stats = new_stats
        
        return reward

    def _traffic_generator(self, timestamp):
        """(内部) 流量生成器"""
        for uid in range(self.user_num):
            for ac_num in range(3):
                if self.rate_table[uid][ac_num][0] > 0:
                    if timestamp % self.rate_table[uid][ac_num][1] == 0:
                        if (self.tx_info_table[uid][ac_num]["write_ptr"] + 1) % self.queue_size == self.tx_info_table[uid][ac_num]["read_ptr"]:
                            continue
                        packet_size = np.random.poisson(lam=self.rate_table[uid][ac_num][0] * 0.001)
                        pkt_info = {'size': packet_size, 'timestamp': timestamp}
                        
                        self.tx_info_table[uid][ac_num]["tot_size"] += packet_size
                        w_ptr = self.tx_info_table[uid][ac_num]["write_ptr"]
                        self.txq_table[uid][ac_num][w_ptr] = pkt_info
                        self.tx_info_table[uid][ac_num]["write_ptr"] = (w_ptr + 1) % self.queue_size
                        
        heapq.heappush(self.event_queue, Task("traffic_generate", timestamp + 1))

    def _handle_timeouts_and_get_lists(self, timestamp):
        """(内部) 清理超时包，并返回待调度列表"""
        vo_list, vi_list, be_list = [], [], []
        for uid in range(self.user_num):
            for ac in range(3):
                info = self.tx_info_table[uid][ac]
                r_ptr = info["read_ptr"]
                w_ptr = info["write_ptr"]
                
                # 1. 丢弃超时包
                timeout = [20, 50, -1][ac] # BE (ac=2) 永不超时
                
                while r_ptr != w_ptr:
                    pkt_timestamp = self.txq_table[uid][ac][r_ptr]['timestamp']
                    age = timestamp - pkt_timestamp
                    
                    if ac == 0 and age > 20: # VO 超时
                        pkt_size = self.txq_table[uid][ac][r_ptr]['size']
                        statistics['total_packets_dropped'][ac] += 1
                        statistics['total_bytes_dropped'][ac] += pkt_size
                        info["tot_size"] -= pkt_size
                        r_ptr = (r_ptr + 1) % self.queue_size
                    elif ac == 1 and age > 50: # VI 超时
                        pkt_size = self.txq_table[uid][ac][r_ptr]['size']
                        statistics['total_packets_dropped'][ac] += 1
                        statistics['total_bytes_dropped'][ac] += pkt_size
                        info["tot_size"] -= pkt_size
                        r_ptr = (r_ptr + 1) % self.queue_size
                    else:
                        break # 找到第一个未超时的包
                
                info["read_ptr"] = r_ptr # 更新读指针

                # 2. 如果队列非空，加入待选列表
                if r_ptr != w_ptr:
                    pkt_info = self.txq_table[uid][ac][r_ptr]
                    schedule_info = {
                        "uid": uid, "ac_type": ac,
                        "size": info["tot_size"],
                        "timestamp": pkt_info['timestamp']
                    }
                    if ac == 0: vo_list.append(schedule_info)
                    elif ac == 1: vi_list.append(schedule_info)
                    else: be_list.append(schedule_info)
        
        return vo_list, vi_list, be_list

    def _run_traditional_scheduler(self, vo, vi, be, timestamp):
        """(内部) 运行一次传统调度逻辑"""
        sorted_vo_list = sorted(vo, key=cmp_to_key(cmp))
        for item in sorted_vo_list:
            if timestamp - item["timestamp"] <= 20:
                return {"uid": item['uid'], "ac_type": 0}
        
        sorted_vi_list = sorted(vi, key=cmp_to_key(cmp))
        for item in sorted_vi_list:
            if 20 < timestamp - item["timestamp"] <= 50:
                 return {"uid": item['uid'], "ac_type": 1}

        sorted_be_list = sorted(be, key=cmp_to_key(cmp))
        for item in sorted_be_list:
            if timestamp - item["timestamp"] > 100:
                return {"uid": item['uid'], "ac_type": 2}
        
        # RR 调度
        max_pkt_size = 0
        res_uid, res_ac_type = -1, -1
        all_lists = sorted_vi_list + sorted_be_list # 优先 VI
        for item in all_lists:
             if item["size"] > max_pkt_size:
                max_pkt_size = item["size"]
                res_uid = item["uid"]
                res_ac_type = item["ac_type"]
                
        return {"uid": res_uid, "ac_type": res_ac_type}

    def _execute_schedule(self, sched_uid, sched_tid, timestamp):
        """(内部) 执行调度，发送数据包，返回发送耗时"""
        if sched_uid == -1:
            return 1.0 # 没调度，也消耗 1ms
            
        info = self.tx_info_table[sched_uid][sched_tid]
        r_ptr = info["read_ptr"]
        w_ptr = info["write_ptr"]
        
        # 1ms 内最多发送的字节
        max_send_bytes = self.bytes_per_ms * 1.0 
        
        temp_sched_size = 0
        
        while r_ptr != w_ptr:
            packet = self.txq_table[sched_uid][sched_tid][r_ptr]
            packet_size = packet["size"]
            
            if temp_sched_size + packet_size <= max_send_bytes:
                # 完整发送这个包
                temp_sched_size += packet_size
                
                latency = timestamp - packet["timestamp"]
                statistics['total_latency'][sched_tid] += latency
                statistics['total_packets_sent'][sched_tid] += 1
                
                r_ptr = (r_ptr + 1) % self.queue_size
            else:
                # 包太大，只能部分发送
                remaining_bytes_to_send = max_send_bytes - temp_sched_size
                if remaining_bytes_to_send > 0:
                    packet["size"] = packet_size - remaining_bytes_to_send
                    info["tot_size"] -= remaining_bytes_to_send
                    temp_sched_size += remaining_bytes_to_send
                    
                    latency = timestamp - packet["timestamp"]
                    statistics['total_latency'][sched_tid] += latency
                    statistics['total_packets_sent'][sched_tid] += 1
                break # 达到最大发送量，跳出
            
        info["read_ptr"] = r_ptr
        info["tot_size"] -= temp_sched_size
        statistics['total_bytes_sent'][sched_tid] += temp_sched_size
        
        # 假设总是固定的 1ms 调度间隔
        return 1.0

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 1. 重置仿真状态
        self.txq_table = create_txq_table(self.user_num, self.queue_size)
        self.tx_info_table = create_tx_info_table(self.user_num)
        self.rate_table = create_rate_table(self.user_num)
        
        self.current_time = 0.0
        self.event_queue = []
        heapq.heappush(self.event_queue, Task("traffic_generate", 0.0))
        heapq.heappush(self.event_queue, Task("schedule", 0.5))
        
        # 2. 重置统计
        reset_statistics()
        self.last_step_stats = self._get_current_stats()
        
        # 3. 运行到第一个调度点
        while self.event_queue:
            task = heapq.heappop(self.event_queue)
            self.current_time = task.timestamp
            
            if task.event == "traffic_generate":
                self._traffic_generator(task.timestamp)
            elif task.event == "schedule":
                # 到达第一个决策点，停止并等待 agent 动作
                heapq.heappush(self.event_queue, task) # 把调度事件放回去
                break
        
        info = {}
        return self._get_obs(), info

    def step(self, action):
        """执行一个 PPO 步骤"""
        
        # 1. 确定此步骤的结束时间
        step_end_time = self.current_time + self.step_duration
        
        # 2. 运行事件循环，直到a)到达调度点 或 b)到达步骤结束时间
        while self.event_queue:
            task = heapq.heappop(self.event_queue)
            
            if task.timestamp >= step_end_time:
                # 达到 step 持续时间，停止运行
                heapq.heappush(self.event_queue, task) # 把事件放回去
                self.current_time = step_end_time
                break
            
            self.current_time = task.timestamp
            
            if task.event == "traffic_generate":
                self._traffic_generator(task.timestamp)
            
            elif task.event == "schedule":
                # --- PPO 决策点 ---
                
                # a. 清理超时包
                self._handle_timeouts_and_get_lists(self.current_time)
                
                # b. 解析 PPO 动作
                if action == self.n_actions - 1: # No-Op 动作
                    sched_uid, sched_tid = -1, -1
                else:
                    sched_uid = action // 3
                    sched_tid = action % 3
                    # 检查动作是否有效（队列是否为空）
                    if self.tx_info_table[sched_uid][sched_tid]["tot_size"] == 0:
                        sched_uid, sched_tid = -1, -1 # 无效动作，等同于 No-Op
                
                # c. 执行调度并获取发送时间
                send_time_ms = self._execute_schedule(sched_uid, sched_tid, self.current_time)
                
                # d. 添加下一个调度事件
                heapq.heappush(self.event_queue, Task("schedule", self.current_time + send_time_ms))
                
                # PPO step 只做一次决策，然后就返回
                # （这是一种常见的 RL + 仿真 结合方式）
                break
        
        # 3. 计算奖励
        reward = self._calculate_reward()
        
        # 4. 检查是否结束
        done = self.current_time >= self.total_sim_duration
        
        # 5. 获取新状态
        obs = self._get_obs()
        
        truncated = False
        info = {}
        
        return obs, reward, done, truncated, info


# --- 4. PPO 代理和网络 ---

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """初始化神经网络层"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    """PPO 代理，包含 Actor 和 Critic 网络"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # Critic (价值网络)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )
        
        # Actor (策略网络)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01)
        )

    def get_value(self, x):
        """获取状态 x 的价值"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """获取动作、动作的 log-prob、熵 和 状态价值"""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x)
        
        return action, log_prob, entropy, value

# --- 5. Rollout 缓冲区 ---
class RolloutBuffer:
    """用于存储 PPO 训练数据"""
    def __init__(self, n_steps, obs_dim, device, gamma, gae_lambda):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.n_steps, self.obs_dim)).to(self.device)
        self.actions = torch.zeros(self.n_steps).to(self.device)
        self.log_probs = torch.zeros(self.n_steps).to(self.device)
        self.rewards = torch.zeros(self.n_steps).to(self.device)
        self.dones = torch.zeros(self.n_steps).to(self.device)
        self.values = torch.zeros(self.n_steps).to(self.device)
        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.step] = torch.tensor(obs).to(self.device)
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step = (self.step + 1) % self.n_steps

    def compute_returns_and_advantage(self, last_value, last_done):
        """计算 GAE (Generalized Advantage Estimation)"""
        with torch.no_grad():
            advantages = torch.zeros(self.n_steps).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    next_non_terminal = 1.0 - last_done
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_value = self.values[t + 1]
                
                delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            returns = advantages + self.values
            return advantages, returns

    def get_batch(self):
        return self.obs, self.actions, self.log_probs, self.rewards, self.dones, self.values


# --- 6. 训练和评估主程序 ---

if __name__ == "__main__":
    
    # --- PPO 超参数 ---
    USER_NUM = 8                 # (调小) 用户数，原为 32。数量越多，状态空间越大，训练越慢
    QUEUE_SIZE = 64              # (调小) 队列大小，原为 512
    SIM_DURATION_MS = 10000      # (调小) 每个 episode 的仿真时长 (10 秒)
    STEP_DURATION_MS = 10        # 每个 PPO step 仿真 10ms
    
    TOTAL_TRAINING_TIMESTEPS = 10000 # (调小) 总训练步数，用于快速演示
    N_STEPS = 128                # PPO 缓冲区大小
    BATCH_SIZE = 64
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    LR = 2.5e-4
    ENT_COEF = 0.01
    VF_COEF = 0.5
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # --- 1. PPO 训练 ---
    print("\n--- 开始 PPO 训练 ---")
    
    # 初始化环境
    env = SchedulerEnv(USER_NUM, QUEUE_SIZE, SIM_DURATION_MS, STEP_DURATION_MS)
    
    # 初始化 PPO 代理
    agent = PPOAgent(env.obs_dim, env.n_actions).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    # 初始化缓冲区
    buffer = RolloutBuffer(N_STEPS, env.obs_dim, DEVICE, GAMMA, GAE_LAMBDA)
    
    global_step = 0
    start_time = time.time()
    
    next_obs, _ = env.reset()
    next_done = torch.zeros(1, dtype=torch.float32).to(DEVICE)
    
    num_updates = TOTAL_TRAINING_TIMESTEPS // N_STEPS
    
    for update in range(1, num_updates + 1):
        
        for step in range(N_STEPS):
            global_step += 1
            
            # 1. 采样动作
            with torch.no_grad():
                obs_tensor = torch.tensor(next_obs).to(DEVICE)
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)
            
            # 2. 执行动作
            next_obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
            reward = torch.tensor(reward).to(DEVICE).view(-1)
            next_done = torch.tensor(done, dtype=torch.float32).to(DEVICE)
            
            # 3. 存储数据
            buffer.add(obs_tensor.cpu().numpy(), action, log_prob, reward, next_done, value.view(-1))
            
            if done or truncated:
                next_obs, _ = env.reset()
                next_done = torch.zeros(1, dtype=torch.float32).to(DEVICE)
        
        # 4. PPO 更新
        with torch.no_grad():
            next_value = agent.get_value(torch.tensor(next_obs).to(DEVICE)).reshape(1, -1)
            advantages, returns = buffer.compute_returns_and_advantage(next_value, next_done)
        
        b_obs, b_actions, b_log_probs, b_rewards, b_dones, b_values = buffer.get_batch()
        
        b_inds = np.arange(N_STEPS)
        for epoch in range(N_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, N_STEPS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]
                
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = torch.exp(log_ratio)
                
                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # PPO 裁剪损失
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 价值损失
                v_loss = F.mse_loss(new_value.view(-1), returns[mb_inds])
                
                # 熵损失 (鼓励探索)
                entropy_loss = entropy.mean()
                
                # 总损失
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        print(f"Update {update}/{num_updates}, Global Step {global_step}, SPS: {int(global_step / (time.time() - start_time))}")
    
    print(f"训练完成, 耗时: {time.time() - start_time:.2f} 秒")
    
    
    # --- 2. 最终评估 ---
    
    EVAL_DURATION_MS = 50000 # 评估 50 秒
    
    # --- 评估 A: 传统调度器 ---
    print("\n--- 正在评估: 传统调度器 ---")
    reset_statistics()
    eval_env = SchedulerEnv(USER_NUM, QUEUE_SIZE, EVAL_DURATION_MS, 1) # step=1ms
    obs, _ = eval_env.reset()
    done = False
    
    while not done:
        # **在这里，我们不使用 PPO 动作，而是调用传统调度逻辑**
        
        # 1. 清理超时并获取列表
        vo, vi, be = eval_env._handle_timeouts_and_get_lists(eval_env.current_time)
        
        # 2. 运行传统调度器
        sched_res = eval_env._run_traditional_scheduler(vo, vi, be, eval_env.current_time)
        
        # 3. 将传统决策 (dict) 转换为 PPO 动作 (int)
        #    (这一步只是为了能调用 env.step()，实际上 step 里的 PPO 逻辑会被忽略)
        #    我们将在 step 内部“劫持”这个决策
        #    为了简单起见... 我们直接调用内部函数
        
        send_time = eval_env._execute_schedule(sched_res["uid"], sched_res["ac_type"], eval_env.current_time)
        eval_env.current_time += send_time
        
        # 手动驱动流量生成器 (因为我们没有调用 step)
        if "next_traffic_gen" not in locals() or eval_env.current_time >= next_traffic_gen:
            eval_env._traffic_generator(eval_env.current_time)
            # 找到下一个流量生成事件
            next_traffic_gen = float('inf')
            for task in eval_env.event_queue:
                if task.event == "traffic_generate":
                    next_traffic_gen = task.timestamp
                    break
        
        done = eval_env.current_time >= EVAL_DURATION_MS

    print_statistics("Traditional", EVAL_DURATION_MS)


    # --- 评估 B: 训练好的 PPO 调度器 ---
    print("\n--- 正在评估: PPO 调度器 (已训练) ---")
    reset_statistics()
    eval_env = SchedulerEnv(USER_NUM, QUEUE_SIZE, EVAL_DURATION_MS, 10) # step=10ms
    obs, _ = eval_env.reset()
    done = False
    
    agent.eval() # 设置为评估模式
    
    while not done:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).to(DEVICE)
            # **这次不采样，而是取概率最高的动作 (确定性策略)**
            logits = agent.actor(obs_tensor)
            action = torch.argmax(logits, dim=-1)
        
        obs, reward, done, truncated, _ = eval_env.step(action.cpu().numpy())
        
    print_statistics("PPO (Trained)", EVAL_DURATION_MS)

    print("\n--- 评估完成 ---")