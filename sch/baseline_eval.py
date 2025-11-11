
# baseline_eval.py
# Run the original 'traditional scheduling' simulator from simcore.py under a fixed seed
# and compute summary metrics that are comparable to the RL environment.
# Author: ChatGPT

import os
import sys
import math
import numpy as np
import random
from typing import Dict, Any, Tuple, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import simcore  # noqa
import heapq

def _weighted_max_hol_delay(txq_table, tx_info_table, user_num, now_ms, weights=(1.0, 0.5, 0.2)) -> float:
    max_delays = [0.0, 0.0, 0.0]
    for uid in range(user_num):
        for ac in range(3):
            r_ptr = tx_info_table[uid][ac]["read_ptr"]
            w_ptr = tx_info_table[uid][ac]["write_ptr"]
            if r_ptr != w_ptr:
                hol_ts = txq_table[uid][ac][r_ptr]["timestamp"]
                hol_delay = max(0.0, now_ms - hol_ts)
                if hol_delay > max_delays[ac]:
                    max_delays[ac] = hol_delay
    return weights[0]*max_delays[0] + weights[1]*max_delays[1] + weights[2]*max_delays[2]

def run_traditional(seed: int = 42,
                    user_num: int = 32,
                    queue_size: int = 512,
                    duration_per_time: int = 10000,
                    runtimes: int = 1,
                    weights=(1.0, 0.5, 0.2)) -> Dict[str, Any]:
    # Ensure deterministic randomness like RL run
    random.seed(seed)
    np.random.seed(seed)

    txq_table = simcore.create_txq_table(user_num, queue_size)
    tx_info_table = simcore.create_tx_info_table(user_num)
    rate_table = simcore.create_rate_table(user_num)

    # Priority queue of events
    pq = []
    heapq.heappush(pq, simcore.Task("traffic_generate", 0))
    heapq.heappush(pq, simcore.Task("schedule", 0.5))

    end_ms = runtimes * duration_per_time

    # Metrics
    weighted_delays = []
    sched_events = 0

    while True:
        task = heapq.heappop(pq)
        t = task.timestamp

        if task.event == "traffic_generate":
            simcore.traffic_generator(txq_table, user_num, int(math.floor(t)), rate_table, tx_info_table, queue_size, pq)
        else:
            # Before applying schedule at time t, compute the (pre-schedule) weighted HOL delay statistic
            wd = _weighted_max_hol_delay(txq_table, tx_info_table, user_num, t, weights=weights)
            weighted_delays.append(wd)
            sched_events += 1
            simcore.schedule(txq_table, tx_info_table, user_num, t, queue_size, pq)

        # Rate table rollover
        if math.floor(t) % duration_per_time == 0 and t > 0:
            rate_table = simcore.create_rate_table(user_num)

        if t > end_ms:
            break

    import numpy as _np
    result = {
        "seed": seed,
        "sched_events": sched_events,
        "mean_weighted_pre_sched_delay": float(_np.mean(weighted_delays)) if weighted_delays else 0.0,
        "p95_weighted_pre_sched_delay": float(_np.percentile(weighted_delays, 95)) if weighted_delays else 0.0,
    }
    return result


def main():
    out = run_traditional(seed=42, user_num=32, queue_size=512, duration_per_time=10000, runtimes=1)
    print(out)

if __name__ == "__main__":
    main()
