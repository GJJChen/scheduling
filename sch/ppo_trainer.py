
# ppo_trainer.py
# Minimal PPO implementation with masked discrete actions, vectorized envs, tqdm progress bars,
# multi-core (Subproc) envs, and optional multi-GPU via torch.nn.DataParallel.
# Author: ChatGPT

import os
import sys
import time
import math
import json
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm

# Local imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from rl_env import SchedulingEnv, EnvConfig  # noqa

# --------- Utilities for vectorized envs (multi-core) ---------
import multiprocessing as mp

def _worker(remote, parent_remote, env_fn_wrapped):
    parent_remote.close()
    env: SchedulingEnv = env_fn_wrapped()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send((obs, info))
            elif cmd == "step":
                obs, reward, done, trunc, info = env.step(data)
                if done:
                    obs, info = env.reset(seed=np.random.randint(0, 1<<31))
                remote.send((obs, reward, done, trunc, info))
            elif cmd == "get_spaces":
                # observation and action dims
                dummy_obs, info = env.reset(seed=np.random.randint(0, 1<<31))
                remote.send((dummy_obs.shape[0], env.action_dim))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        pass

class VecEnv:
    def __init__(self, env_fns: List[Callable[[], SchedulingEnv]], seeds: Optional[List[int]] = None):
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.ps = [mp.Process(target=_worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for work_remote in self.work_remotes:
            work_remote.close()

        # get spaces
        self.remotes[0].send(("get_spaces", None))
        obs_dim, action_dim = self.remotes[0].recv()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # reset all
        for i, r in enumerate(self.remotes):
            seed = seeds[i] if seeds is not None else np.random.randint(0, 1<<31)
            r.send(("reset", seed))

        obs_info = [r.recv() for r in self.remotes]
        self.obs = np.stack([x[0] for x in obs_info], axis=0)
        self.infos = [x[1] for x in obs_info]

    def step(self, actions: np.ndarray):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        results = [r.recv() for r in self.remotes]
        obs, rewards, dones, truncs, infos = zip(*results)
        self.obs = np.stack(obs, axis=0)
        self.infos = list(infos)
        return self.obs, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=np.bool_), infos

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.ps:
            p.join()

# --------- PPO Agent ---------
class MaskedCategorical(Categorical):
    """
    Categorical distribution with action masking by setting logits of invalid actions to -inf.
    """
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            # mask shape: [B, A]; logits shape: [B, A]
            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
        super().__init__(logits=logits)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(512, 256)):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy_net(x)
        value = self.value_net(x).squeeze(-1)
        return logits, value

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

@dataclass
class PPOConfig:
    total_episodes: int = 200
    n_envs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 2048
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpu_ids: Optional[List[int]] = field(default_factory=lambda: [0,1,2,3])  # e.g. [0,1] to enable DataParallel
    seed: int = 42
    log_dir: str = os.path.join(THIS_DIR, "logs")
    model_path: str = os.path.join(THIS_DIR, "ppo_model.pt")

class PPOTrainer:
    def __init__(self, env_cfg: EnvConfig, ppo_cfg: PPOConfig):
        self.env_cfg = env_cfg
        self.cfg = ppo_cfg
        os.makedirs(self.cfg.log_dir, exist_ok=True)

        # Vectorized envs with different seeds for decorrelation
        def make_env(s):
            def _thunk():
                ec = EnvConfig(**vars(self.env_cfg))
                ec.seed = s
                return SchedulingEnv(ec)
            return _thunk

        seeds = [self.cfg.seed + i*13 for i in range(self.cfg.n_envs)]
        self.venv = VecEnv([make_env(s) for s in seeds], seeds=seeds)

        self.obs_dim = self.venv.obs.shape[1]
        self.action_dim = self.venv.action_dim

        self.device = torch.device(self.cfg.device)
        self.model = ActorCritic(self.obs_dim, self.action_dim).to(self.device)

        # Optional multi-GPU support
        if self.cfg.multi_gpu_ids is not None and len(self.cfg.multi_gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.cfg.multi_gpu_ids)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

        # Logging
        self.train_csv = os.path.join(self.cfg.log_dir, "train.csv")
        with open(self.train_csv, "w") as f:
            f.write("episode,mean_return")

    def _collect_rollout(self, steps_per_episode: int) -> Dict[str, Any]:
        """
        Collect one episode worth of experience from each env.
        We terminate an episode when the underlying env signals done.
        """
        obs_list, actions, logprobs, rewards, dones, values, masks = [], [], [], [], [], [], []

        # We'll accumulate at least 'steps_per_episode' steps per env, but will break on 'done'
        t_steps = 0
        while t_steps < steps_per_episode:
            obs_np = self.venv.obs  # [n_envs, obs_dim]
            infos = self.venv.infos
            action_masks = np.stack([info["action_mask"] for info in infos], axis=0)  # [n_envs, A]

            obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

            with torch.no_grad():
                logits, value = self.model(obs)
                dist = MaskedCategorical(logits=logits, mask=mask_t)
                action = dist.sample()
                logprob = dist.log_prob(action)

            next_obs, r, d, _infos = self.venv.step(action.cpu().numpy())

            obs_list.append(obs_np)
            actions.append(action.cpu().numpy())
            logprobs.append(logprob.cpu().numpy())
            rewards.append(r)
            dones.append(d.astype(np.float32))
            values.append(value.cpu().numpy())
            masks.append(action_masks)

            t_steps += 1

        # Convert to arrays
        data = {
            "obs": np.asarray(obs_list, dtype=np.float32),               # [T, N, obs_dim]
            "actions": np.asarray(actions, dtype=np.int64),              # [T, N]
            "logprobs": np.asarray(logprobs, dtype=np.float32),          # [T, N]
            "rewards": np.asarray(rewards, dtype=np.float32),            # [T, N]
            "dones": np.asarray(dones, dtype=np.float32),                # [T, N]
            "values": np.asarray(values, dtype=np.float32),              # [T, N]
            "masks": np.asarray(masks, dtype=np.float32),                # [T, N, A]
        }
        return data

    def _compute_gae(self, data: Dict[str, Any], last_value: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE and returns over the rollout. We assume episodes end naturally; for simplicity, we bootstrap with 0.
        """
        rewards = data["rewards"]            # [T, N]
        values = data["values"]              # [T, N]
        dones = data["dones"]                # [T, N]
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros((N,), dtype=np.float32)
        # We bootstrap with value=0 at terminal for simplicity
        for t in reversed(range(T)):
            next_values = values[t+1] if t+1 < T else np.zeros((N,), dtype=np.float32)
            delta = rewards[t] + self.cfg.gamma * next_values * (1.0 - dones[t]) - values[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def _update(self, data: Dict[str, Any]):
        T, N = data["rewards"].shape
        obs = torch.tensor(data["obs"].reshape(T*N, -1), dtype=torch.float32, device=self.device)
        actions = torch.tensor(data["actions"].reshape(T*N), dtype=torch.int64, device=self.device)
        old_logprobs = torch.tensor(data["logprobs"].reshape(T*N), dtype=torch.float32, device=self.device)
        masks = torch.tensor(data["masks"].reshape(T*N, -1), dtype=torch.bool, device=self.device)

        adv, ret = self._compute_gae(data)
        advantages = torch.tensor(adv.reshape(T*N), dtype=torch.float32, device=self.device)
        returns = torch.tensor(ret.reshape(T*N), dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_inds = np.arange(T*N)
        batch_size = T*N
        for epoch in range(self.cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = b_inds[start:end]

                logits, values = self.model(obs[mb_inds])
                dist = MaskedCategorical(logits=logits, mask=masks[mb_inds])
                logprobs = dist.log_prob(actions[mb_inds])
                entropy = dist.entropy().mean()

                ratio = (logprobs - old_logprobs[mb_inds]).exp()
                pg_loss1 = -advantages[mb_inds] * ratio
                pg_loss2 = -advantages[mb_inds] * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (returns[mb_inds] - values).pow(2).mean()
                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

    def train(self, steps_per_episode: int = 2048):
        ep_bar = tqdm(range(self.cfg.total_episodes), desc="Training (episodes)")
        mean_returns = []
        for ep in ep_bar:
            data = self._collect_rollout(steps_per_episode)
            # Mean episode return across envs for reporting
            mean_ret = float(np.mean(np.sum(data["rewards"], axis=0)))
            mean_returns.append(mean_ret)
            with open(self.train_csv, "a") as f:
                f.write(f"{ep},{mean_ret}")
            ep_bar.set_postfix({"mean_return": f"{mean_ret:.3f}"})

            self._update(data)

        # Save model
        torch.save(self.model.state_dict(), self.cfg.model_path)

        # Save convergence plot
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(mean_returns)
            plt.xlabel("Episode")
            plt.ylabel("Mean episodic return")
            plt.title("PPO Reward Convergence")
            plt.grid(True)
            out_path = os.path.join(self.cfg.log_dir, "reward_curve.png")
            plt.savefig(out_path, dpi=140, bbox_inches="tight")
        except Exception as e:
            print("Plot failed:", e)

    def close(self):
        self.venv.close()


def main():
    # Example default configs
    env_cfg = EnvConfig(user_num=32, queue_size=512, duration_per_time=10000, runtimes=1,
                        weights=(1.0, 0.6, 0.3), seed=42)
    ppo_cfg = PPOConfig(total_episodes=10, n_envs=2, seed=42)
    trainer = PPOTrainer(env_cfg, ppo_cfg)
    try:
        trainer.train(steps_per_episode=512)
    finally:
        trainer.close()

if __name__ == "__main__":
    main()
