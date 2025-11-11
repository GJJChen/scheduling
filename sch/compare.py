
# compare.py
# Load a trained PPO policy and compare it against the traditional scheduler on the same seeds.
# Author: ChatGPT

import os
import sys
import numpy as np
import torch
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from rl_env import SchedulingEnv, EnvConfig
from ppo_trainer import ActorCritic
from baseline_eval import run_traditional

def eval_rl(model_path: str, seed: int, env_cfg: EnvConfig) -> dict:
    env = SchedulingEnv(env_cfg)
    obs, info = env.reset(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim=obs.shape[0], action_dim=env.action_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    weighted_delays = []
    done = False
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(info["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
            logits, _ = model(obs_t)
            # Mask invalid actions
            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(mask_t, logits, torch.full_like(logits, neg_inf))
            action = torch.argmax(logits, dim=-1).item()
            # Collect the *pre-schedule* delay to match baseline metric
            wd_pre = env._weighted_max_hol_delay()
            obs, reward, done, _, info = env.step(action)
            weighted_delays.append(wd_pre)

    res = {
        "seed": seed,
        "mean_weighted_pre_sched_delay": float(np.mean(weighted_delays)) if weighted_delays else 0.0,
        "p95_weighted_pre_sched_delay": float(np.percentile(weighted_delays, 95)) if weighted_delays else 0.0,
    }
    return res

def main():
    # Example usage: compare on seeds 42..44
    env_cfg = EnvConfig(user_num=32, queue_size=512, duration_per_time=10000, runtimes=1, weights=(1.0, 0.6, 0.3))
    model_path = os.path.join(THIS_DIR, "ppo_model.pt")

    seeds = [42, 43, 44]
    rows = []
    for s in seeds:
        base = run_traditional(seed=s, user_num=env_cfg.user_num, queue_size=env_cfg.queue_size,
                               duration_per_time=env_cfg.duration_per_time, runtimes=env_cfg.runtimes,
                               weights=env_cfg.weights)
        rl = eval_rl(model_path, seed=s, env_cfg=env_cfg)
        row = {"seed": s,
               "base_mean": base["mean_weighted_pre_sched_delay"],
               "rl_mean": rl["mean_weighted_pre_sched_delay"],
               "base_p95": base["p95_weighted_pre_sched_delay"],
               "rl_p95": rl["p95_weighted_pre_sched_delay"]}
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df)
    out_csv = os.path.join(THIS_DIR, "compare.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
