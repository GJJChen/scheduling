
# rl_env.py
# Custom Gym-like environment wrapping the user's simcore.py simulator for PPO training.
# It DOES NOT modify simcore.py; it only imports and reuses its functions. 
# Reward: negative weighted maximum head-of-line delay across queues after each schedule.
# Author: ChatGPT

import os
import sys
import math
import heapq
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

# Ensure we can import the user's simcore.py in the same folder
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import simcore  # noqa: this is the original simulator provided by the user

@dataclass
class EnvConfig:
    user_num: int = 32
    queue_size: int = 512
    duration_per_time: int = 10000  # ms
    runtimes: int = 1               # number of duration blocks per episode
    max_rate_bps: int = 2442 * 1000 * 1000  # matches simcore
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.2)  # VO, VI, BE
    seed: int = 42
    normalize_obs: bool = True


class SchedulingEnv:
    """
    Light Gym-style API:
      - reset(seed=None) -> obs, info
      - step(action) -> obs, reward, terminated, truncated, info
    Observation: flattened array, for each (uid, ac) we include [tot_size_bits, hol_delay_ms].
    Action: integer in [0, user_num * 3 - 1] => choose (uid, ac) to schedule next.
    A mask of legal actions (non-empty queues) is provided via info['action_mask'].
    """
    metadata = {"render.modes": []}

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.rng_py = random.Random(self.cfg.seed)
        self.rng_np = np.random.RandomState(self.cfg.seed)

        self.user_num = self.cfg.user_num
        self.queue_size = self.cfg.queue_size
        self.action_dim = self.user_num * 3

        self._build_normalizers()

        # State variables (allocated on reset)
        self.txq_table = None
        self.tx_info_table = None
        self.rate_table = None
        self.t_ms = 0.0
        self.ep_end_ms = None

        # For metrics
        self.total_scheduled_bytes = 0
        self.drop_vi = 0
        self.drop_vo = 0

        # Book-keeping for BE min-interval (not enforced, but can be used for features if desired)
        self.last_be_sched_ts = None

    def _build_normalizers(self):
        # Heuristic scales for observation normalization
        # total size can be large; we approximate an upper bound per queue
        # Max per 4ms slot is: rate_bps/1000*4; over 100ms it could pile up
        cap_bits = int(self.cfg.max_rate_bps / 1000.0 * 100.0)
        self.size_scale = max(1.0, cap_bits)
        self.delay_scale = 1000.0  # ms

    # --------------- Public API ---------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.cfg.seed = int(seed)
        # Sync seeds for both python and numpy (and for simcore's random usage)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.rng_py.seed(self.cfg.seed)
        self.rng_np.seed(self.cfg.seed)

        # Allocate simulator structures
        self.txq_table = simcore.create_txq_table(self.cfg.user_num, self.queue_size)
        self.tx_info_table = simcore.create_tx_info_table(self.cfg.user_num)
        self.rate_table = simcore.create_rate_table(self.cfg.user_num)

        self.t_ms = 0.0
        self.ep_end_ms = self.cfg.runtimes * self.cfg.duration_per_time
        self.total_scheduled_bytes = 0
        self.drop_vi = 0
        self.drop_vo = 0
        self.last_be_sched_ts = [[-1.0 for _ in range(3)] for _ in range(self.cfg.user_num)]

        # Prime with first traffic burst and first schedule tick (like simcore main)
        # In the simcore main, traffic_generate starts at t=0 and then repeats every 1ms.
        # We'll call traffic_generator on t=0.
        self._traffic_generate_at(int(math.floor(self.t_ms)))  # t = 0

        obs, info = self._make_obs()
        return obs, info

    def step(self, action: int):
        # If all queues are empty, advance time by 1ms and generate traffic
        if self._all_queues_empty():
            # Idle step: advance one millisecond
            self.t_ms += 1.0
            self._rate_table_rollover_if_needed(int(math.floor(self.t_ms)))
            self._traffic_generate_at(int(math.floor(self.t_ms)))
            obs, info = self._make_obs()
            reward = 0.0
            done = self.t_ms >= self.ep_end_ms
            return obs, reward, done, False, info

        # Apply deadline-based drops using simcore's gen_sched_res side-effects
        # We call it purely to trigger its internal VO/VI drop behavior at the current time.
        _ = simcore.gen_sched_res(self.txq_table, self.tx_info_table, self.cfg.user_num, self.t_ms, self.queue_size)

        # Decode action
        uid, ac = divmod(action, 3)

        # If the selected queue is empty, treat as no-op and small penalty to discourage invalid selections
        if self.tx_info_table[uid][ac]["read_ptr"] == self.tx_info_table[uid][ac]["write_ptr"]:
            # advance time by 1ms (wasted slot)
            self.t_ms += 1.0
            self._rate_table_rollover_if_needed(int(math.floor(self.t_ms)))
            self._traffic_generate_at(int(math.floor(self.t_ms)))
            obs, info = self._make_obs()
            reward = -0.01  # mild penalty for scheduling an empty queue
            done = self.t_ms >= self.ep_end_ms
            return obs, reward, done, False, info

        # Otherwise, schedule like simcore.schedule but for the chosen (uid, ac)
        temp_sched_size = 0.0
        r_ptr = self.tx_info_table[uid][ac]["read_ptr"]
        w_ptr = self.tx_info_table[uid][ac]["write_ptr"]

        max_quanta_bits = self.cfg.max_rate_bps / 1000.0 * 4.0  # same as simcore (4ms quanta)
        while r_ptr != w_ptr:
            if temp_sched_size == max_quanta_bits:
                break
            pkt_size = self.txq_table[uid][ac][r_ptr]["size"]
            if temp_sched_size + pkt_size <= max_quanta_bits:
                temp_sched_size += pkt_size
                r_ptr = (r_ptr + 1) % self.queue_size
            else:
                # partially send the head packet
                self.txq_table[uid][ac][r_ptr]["size"] = pkt_size - (max_quanta_bits - temp_sched_size)
                temp_sched_size = max_quanta_bits

        self.tx_info_table[uid][ac]["read_ptr"] = r_ptr
        self.tx_info_table[uid][ac]["tot_size"] -= temp_sched_size
        send_time_ms = (temp_sched_size / self.cfg.max_rate_bps) * 1000.0
        if send_time_ms <= 0.0:
            # Safety against rare 0 send
            send_time_ms = 1.0

        # Between now and t + send_time_ms, traffic continues to arrive each ms
        steps = max(1, int(math.ceil(send_time_ms)))
        for _ in range(steps):
            self.t_ms += 1.0
            t_int = int(math.floor(self.t_ms))
            self._rate_table_rollover_if_needed(t_int)
            self._traffic_generate_at(t_int)

        # Reward: negative weighted maximum HOL delay across VO/VI/BE *after* scheduling
        reward = -self._weighted_max_hol_delay()

        obs, info = self._make_obs()
        done = self.t_ms >= self.ep_end_ms
        return obs, reward, done, False, info

    # --------------- Helpers ---------------
    def _all_queues_empty(self) -> bool:
        for uid in range(self.cfg.user_num):
            for ac in range(3):
                if self.tx_info_table[uid][ac]["write_ptr"] != self.tx_info_table[uid][ac]["read_ptr"]:
                    return False
        return True

    def _rate_table_rollover_if_needed(self, t_int: int):
        # Match simcore: every 'duration_per_time' ms, rebuild the rate table with the *same RNG usage*
        if t_int > 0 and (t_int % self.cfg.duration_per_time == 0):
            self.rate_table = simcore.create_rate_table(self.cfg.user_num)

    def _traffic_generate_at(self, t_int: int):
        # Adapted call into simcore's traffic_generator without using its event queue
        # We pass a dummy list for event_queue; the function will push into it but we ignore the entries.
        dummy_eq = []
        simcore.traffic_generator(self.txq_table, self.cfg.user_num, t_int, self.rate_table, self.tx_info_table, self.queue_size, dummy_eq)

    def _weighted_max_hol_delay(self) -> float:
        max_delays = [0.0, 0.0, 0.0]
        for uid in range(self.cfg.user_num):
            for ac in range(3):
                r_ptr = self.tx_info_table[uid][ac]["read_ptr"]
                w_ptr = self.tx_info_table[uid][ac]["write_ptr"]
                if r_ptr != w_ptr:
                    hol_ts = self.txq_table[uid][ac][r_ptr]["timestamp"]
                    hol_delay = max(0.0, self.t_ms - hol_ts)
                    if hol_delay > max_delays[ac]:
                        max_delays[ac] = hol_delay
        weights = self.cfg.weights
        return weights[0]*max_delays[0] + weights[1]*max_delays[1] + weights[2]*max_delays[2]

    def _make_obs(self) -> tuple:
        feats = []
        mask = []
        for uid in range(self.cfg.user_num):
            for ac in range(3):
                r_ptr = self.tx_info_table[uid][ac]["read_ptr"]
                w_ptr = self.tx_info_table[uid][ac]["write_ptr"]
                tot_size = float(self.tx_info_table[uid][ac]["tot_size"])
                if r_ptr != w_ptr:
                    hol_ts = self.txq_table[uid][ac][r_ptr]["timestamp"]
                    hol_delay = max(0.0, self.t_ms - hol_ts)
                    mask.append(1.0)  # can schedule
                else:
                    hol_delay = 0.0
                    mask.append(0.0)  # cannot schedule (empty)

                if self.cfg.normalize_obs:
                    feats.extend([tot_size/self.size_scale, hol_delay/self.delay_scale])
                else:
                    feats.extend([tot_size, hol_delay])

        obs = np.asarray(feats, dtype=np.float32)
        info = {"action_mask": np.asarray(mask, dtype=np.float32),
                "t_ms": self.t_ms}
        return obs, info
