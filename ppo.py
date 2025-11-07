"""
End-to-end wireless scheduler simulation + PPO agent (multi-process) with p95 latency CSV logging
- Matches the problem statement:
  * Users: 1–64, each with VO/VI/BE queues
  * 1 ms time-step; if no buffer, wait 1 ms
  * Traffic models: periodic bytes-per-packet, Poisson arrivals
  * PHY: 2442 Mbps, fixed overhead 150 us per burst, max burst 4 ms
  * Legacy scheduler baseline with the 1→4 rules
  * RL scheduler chooses (which user, which class, how long to transmit)

New in this version:
  * **Multi-process** rollout collection for PPO (--procs N)
  * **p95 latency curves** computed over sliding time windows and **CSV export** (--csv_out path, --csv_window_ms 100)

Run examples (CPU, PyTorch required):

  # Baseline with CSV output every 100 ms
  python scheduler_rl_sim.py --users 16 --minutes 0.2 --mode baseline --csv_out baseline.csv --csv_window_ms 100

  # Train PPO with 4 worker processes and CSV logs during final eval
  python scheduler_rl_sim.py --users 16 --minutes 0.5 --mode train --iters 50 --procs 4 \
      --eval_csv_out rl_eval.csv --csv_window_ms 100

  # Eval a checkpoint and export CSV
  python scheduler_rl_sim.py --users 16 --minutes 0.2 --mode eval --checkpoint ppo_checkpoint.pt --eval_csv_out rl_eval.csv
"""
from __future__ import annotations
import math
import argparse
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp

# =========================
# Constants & Utilities
# =========================
BITS_PER_SEC = 2442e6
OVERHEAD_S = 150e-6
MAX_BURST_S = 4e-3
MS = 1e-3

VO_DROP_MS = 20.0
VI_NEG_MS = 50.0
BE_GUARANTEE_MS = 100.0

DUR_CHOICES_MS = [0.25, 0.5, 1.0, 2.0, 4.0]
DUR_CHOICES_S = [d/1000.0 for d in DUR_CHOICES_MS]

# Observation feature scaling (rough bounds)
MAX_QUEUE_BYTES = 1_000_000   # 1 MB per queue as a soft cap for normalization
MAX_DELAY_MS = 1_000.0

CLASSES = ["VO", "VI", "BE"]
CLASS_IDX = {"VO":0, "VI":1, "BE":2}

# =========================
# Traffic Models
# =========================
@dataclass
class PeriodicTraffic:
    bytes_per_pkt: int
    period_ms: int
    jitter_ms: int = 0
    next_time_ms: int = 0

    def step_arrivals(self, t_ms: int) -> List[int]:
        arrivals = []
        while t_ms >= self.next_time_ms:
            arrivals.append(self.bytes_per_pkt)
            # schedule next
            j = 0
            if self.jitter_ms > 0:
                j = random.randint(-self.jitter_ms, self.jitter_ms)
            self.next_time_ms += max(1, self.period_ms + j)
        return arrivals

@dataclass
class PoissonTraffic:
    lambda_per_ms: float  # expected packets per ms
    bytes_per_pkt: int

    def step_arrivals(self, t_ms: int) -> List[int]:
        k = np.random.poisson(self.lambda_per_ms)
        return [self.bytes_per_pkt] * int(k)

# =========================
# Queue structure
# =========================
@dataclass
class Pkt:
    bytes: int
    t_enqueue_ms: int

@dataclass
class FlowQueue:
    # FIFO of packets
    pkts: List[Pkt] = field(default_factory=list)

    def enqueue(self, bytes_arrivals: List[int], t_ms: int):
        for b in bytes_arrivals:
            self.pkts.append(Pkt(bytes=b, t_enqueue_ms=t_ms))

    def total_bytes(self) -> int:
        return sum(p.bytes for p in self.pkts)

    def head_delay_ms(self, now_ms: int) -> float:
        if not self.pkts:
            return 0.0
        return float(now_ms - self.pkts[0].t_enqueue_ms)

    def drop_older_than(self, now_ms: int, thr_ms: float) -> int:
        # drop all pkts whose head-of-line delay > thr_ms; return count
        dropped = 0
        keep = []
        for p in self.pkts:
            if now_ms - p.t_enqueue_ms > thr_ms:
                dropped += 1
            else:
                keep.append(p)
        self.pkts = keep
        return dropped

    def serve_bits(self, bits_budget: float, now_ms: int) -> Tuple[int, List[float]]:
        """Serve FIFO with possible partial last packet.
        Returns: (bits_sent, full_packet_delays_ms)
        Only when a packet is COMPLETELY transmitted do we record its sojourn delay.
        """
        bytes_budget = int(bits_budget // 8)
        sent = 0
        delays_served: List[float] = []
        new_pkts = []
        for p in self.pkts:
            if bytes_budget <= 0:
                new_pkts.append(p)
                continue
            if p.bytes <= bytes_budget:
                sent += p.bytes
                bytes_budget -= p.bytes
                delays_served.append(float(now_ms - p.t_enqueue_ms))
            else:
                # partial serve
                sent += bytes_budget
                residual = p.bytes - bytes_budget
                new_pkts.append(Pkt(bytes=residual, t_enqueue_ms=p.t_enqueue_ms))
                bytes_budget = 0
        self.pkts = new_pkts
        return sent * 8, delays_served

# =========================
# Environment
# =========================
class SchedulerEnv:
    def __init__(self,
                 n_users: int = 16,
                 minutes: float = 0.2,
                 seed: int = 1,
                 traffic_conf: Optional[Dict] = None,
                 topk: Optional[int] = None,
                 be_mask_window_ms: int = 5,
                 csv_window_ms: Optional[int] = None,
                 csv_path: Optional[str] = None):
        assert 1 <= n_users <= 64
        self.n_users = n_users
        self.steps = int(minutes * 60_000)  # ms steps
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.t_ms = 0
        self.topk = topk
        self.be_mask_window_ms = be_mask_window_ms

        # Build traffic for each user/class
        self.traffic = [[None for _ in CLASSES] for _ in range(n_users)]
        if traffic_conf is None:
            traffic_conf = self.default_traffic_conf()
        for u in range(n_users):
            for ci, cname in enumerate(CLASSES):
                self.traffic[u][ci] = self.make_traffic(traffic_conf[cname])

        # Queues
        self.queues = [[FlowQueue() for _ in CLASSES] for _ in range(n_users)]
        self.be_last_served = np.full((n_users,), -10_000, dtype=np.int32)

        # Stats
        self.neg_events_vo = 0
        self.neg_events_vi = 0
        self.bits_sent = 0
        self.idle_steps = 0

        # Latency samples per class
        self.latency_samples: Dict[str, List[float]] = {c: [] for c in CLASSES}

        # CSV logging
        self.csv_window_ms = csv_window_ms
        self.csv_path = csv_path
        self._csv_rows: List[List] = []
        self._window_samples: Dict[str, List[float]] = {c: [] for c in CLASSES}
        self._next_cut_ms = (csv_window_ms if csv_window_ms else 0)

    def enable_csv(self, csv_window_ms: int, csv_path: str):
        self.csv_window_ms = csv_window_ms
        self.csv_path = csv_path
        self._csv_rows = []
        self._window_samples = {c: [] for c in CLASSES}
        self._next_cut_ms = csv_window_ms

    def reset(self):
        self.t_ms = 0
        self.queues = [[FlowQueue() for _ in CLASSES] for _ in range(self.n_users)]
        self.be_last_served[:] = -10_000
        self.neg_events_vo = 0
        self.neg_events_vi = 0
        self.bits_sent = 0
        self.idle_steps = 0
        self.latency_samples = {c: [] for c in CLASSES}
        self._window_samples = {c: [] for c in CLASSES}
        self._csv_rows = []
        self._next_cut_ms = (self.csv_window_ms if self.csv_window_ms else 0)
        return self._observe()

    @staticmethod
    def default_traffic_conf():
        # Light/moderate mixed traffic defaults
        return {
            "VO": {"type": "poisson", "lambda_per_ms": 0.05, "bytes_per_pkt": 400},
            "VI": {"type": "poisson", "lambda_per_ms": 0.03, "bytes_per_pkt": 1200},
            "BE": {"type": "periodic", "bytes_per_pkt": 1500, "period_ms": 10}
        }

    def make_traffic(self, cfg):
        if cfg["type"] == "periodic":
            return PeriodicTraffic(bytes_per_pkt=cfg["bytes_per_pkt"],
                                   period_ms=cfg.get("period_ms", 10),
                                   jitter_ms=cfg.get("jitter_ms", 0))
        elif cfg["type"] == "poisson":
            return PoissonTraffic(lambda_per_ms=cfg["lambda_per_ms"],
                                  bytes_per_pkt=cfg["bytes_per_pkt"])
        else:
            raise ValueError("unknown traffic type")

    # ---------- Core dynamics ----------
    def step(self, action: Optional[Tuple[int,int]]):
        """
        action: (flat_queue_index, dur_idx)
            flat_queue_index in [0, n_users*3)  (user*3 + class)
            dur_idx in [0, len(DUR_CHOICES_S))
        If action is None or invalid, it's idle this step.
        """
        # 1) Arrivals
        for u in range(self.n_users):
            for ci in range(3):
                arr = self.traffic[u][ci].step_arrivals(self.t_ms)
                self.queues[u][ci].enqueue(arr, self.t_ms)

        # 2) Enforce VO/VI deadlines
        vo_dropped = 0
        for u in range(self.n_users):
            vo_dropped += self.queues[u][CLASS_IDX["VO"]].drop_older_than(self.t_ms, VO_DROP_MS)
        self.neg_events_vo += vo_dropped

        for u in range(self.n_users):
            q = self.queues[u][CLASS_IDX["VI"]]
            if q.pkts and (self.t_ms - q.pkts[0].t_enqueue_ms > VI_NEG_MS):
                self.neg_events_vi += 1

        # 3) BE guarantee past-due detection
        be_due_users = []
        for u in range(self.n_users):
            q = self.queues[u][CLASS_IDX["BE"]]
            if q.pkts:
                if self.t_ms - self.be_last_served[u] >= BE_GUARANTEE_MS:
                    be_due_users.append(u)

        # 4) Serve according to action
        served_bits = 0
        served_delays_by_class: Dict[str, List[float]] = {c: [] for c in CLASSES}
        if action is not None:
            flat_idx, dur_idx = action
            if 0 <= flat_idx < self.n_users*3 and 0 <= dur_idx < len(DUR_CHOICES_S):
                user = flat_idx // 3
                ci = flat_idx % 3
                # Hard override: if there is a BE past due, force serve one of them
                if be_due_users and not (ci == CLASS_IDX["BE"] and user in be_due_users):
                    user = be_due_users[0]
                    ci = CLASS_IDX["BE"]
                duration_s = min(DUR_CHOICES_S[dur_idx], MAX_BURST_S)
                payload_time_s = max(0.0, duration_s - OVERHEAD_S)
                bits_budget = max(0.0, payload_time_s * BITS_PER_SEC)
                bits, delays = self.queues[user][ci].serve_bits(bits_budget, now_ms=self.t_ms)
                served_bits = bits
                cname = CLASSES[ci]
                served_delays_by_class[cname].extend(delays)
                if ci == CLASS_IDX["BE"] and served_bits > 0:
                    self.be_last_served[user] = self.t_ms
        if served_bits == 0:
            self.idle_steps += 1
        self.bits_sent += served_bits

        # Record latency samples for CSV (only full-packet delays)
        for cname in CLASSES:
            if served_delays_by_class[cname]:
                self.latency_samples[cname].extend(served_delays_by_class[cname])
                self._window_samples[cname].extend(served_delays_by_class[cname])

        # CSV window cut
        if self.csv_window_ms is not None and self.csv_window_ms > 0:
            if self.t_ms >= self._next_cut_ms:
                row = self._compute_window_row(end_ms=self._next_cut_ms)
                self._csv_rows.append(row)
                self._window_samples = {c: [] for c in CLASSES}
                self._next_cut_ms += self.csv_window_ms

        # 5) Advance time 1 ms fixed
        self.t_ms += 1
        done = self.t_ms >= self.steps

        # 6) Reward shaping
        reward = self._reward(served_bits)
        obs = self._observe()
        info = {}
        if done and self.csv_path and self._csv_rows:
            self._write_csv()
        return obs, reward, done, info

    def _compute_window_row(self, end_ms: int) -> List:
        def p95(arr):
            return float(np.percentile(arr, 95)) if arr else 0.0
        vo = p95(self._window_samples['VO'])
        vi = p95(self._window_samples['VI'])
        be = p95(self._window_samples['BE'])
        overall = p95(self._window_samples['VO'] + self._window_samples['VI'] + self._window_samples['BE'])
        return [end_ms, vo, vi, be, overall]

    def _write_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_ms','p95_vo_ms','p95_vi_ms','p95_be_ms','p95_overall_ms'])
            w.writerows(self._csv_rows)
        print(f"CSV written: {self.csv_path}  rows={len(self._csv_rows)}")

    def _reward(self, served_bits: int) -> float:
        # delay stats (queue head delays)
        delays_vo = []
        delays_vi = []
        delays_be = []
        for u in range(self.n_users):
            delays_vo.append(self.queues[u][CLASS_IDX["VO"]].head_delay_ms(self.t_ms))
            delays_vi.append(self.queues[u][CLASS_IDX["VI"]].head_delay_ms(self.t_ms))
            delays_be.append(self.queues[u][CLASS_IDX["BE"]].head_delay_ms(self.t_ms))
        avg_vo = np.mean(delays_vo) if delays_vo else 0.0
        avg_vi = np.mean(delays_vi) if delays_vi else 0.0
        avg_be = np.mean(delays_be) if delays_be else 0.0

        viol_vi = 1.0 if any(d > VI_NEG_MS for d in delays_vi) else 0.0

        be_overdue = 0.0
        for u in range(self.n_users):
            q = self.queues[u][CLASS_IDX["BE"]]
            if q.pkts and (self.t_ms - self.be_last_served[u] >= BE_GUARANTEE_MS):
                be_overdue = 1.0
                break

        alpha_vo, alpha_vi, alpha_be = 0.02, 0.01, 0.002
        beta_vi = 1.0
        beta_be = 2.0
        gamma_thr = 1e-9 

        # Final Reward:    
        r = (
            gamma_thr * served_bits
            - alpha_vo * avg_vo
            - alpha_vi * avg_vi
            - alpha_be * avg_be
            - beta_vi * viol_vi
            - beta_be * be_overdue
        )
        return float(r)

    def _observe(self) -> Dict:
        feats = []
        for u in range(self.n_users):
            since = self.t_ms - self.be_last_served[u]
            rem = max(0.0, BE_GUARANTEE_MS - since)
            be_norm = rem / BE_GUARANTEE_MS
            for ci, cname in enumerate(CLASSES):
                q = self.queues[u][ci]
                bytes_norm = min(1.0, q.total_bytes() / MAX_QUEUE_BYTES)
                delay_norm = min(1.0, q.head_delay_ms(self.t_ms) / MAX_DELAY_MS)
                extra = be_norm if cname == "BE" else 0.0
                feats.extend([bytes_norm, delay_norm, extra])
        feats = np.array(feats, dtype=np.float32)

        mask = np.zeros(self.n_users*3*len(DUR_CHOICES_S), dtype=np.float32)
        for u in range(self.n_users):
            for ci, cname in enumerate(CLASSES):
                q = self.queues[u][ci]
                flat = u*3 + ci
                valid_queue = 1 if q.total_bytes() > 0 else 0
                be_due = False
                if cname == "BE" and q.pkts:
                    if self.t_ms - self.be_last_served[u] >= BE_GUARANTEE_MS - self.be_mask_window_ms:
                        be_due = True
                for di in range(len(DUR_CHOICES_S)):
                    idx = flat*len(DUR_CHOICES_S) + di
                    m = valid_queue
                    if be_due:
                        m = 1
                    mask[idx] = m
                if q.pkts and (self.t_ms - self.be_last_served[u] >= BE_GUARANTEE_MS - self.be_mask_window_ms):
                    for other_ci in range(3):
                        if other_ci == CLASS_IDX["BE"]: continue
                        flat2 = u*3 + other_ci
                        for di in range(len(DUR_CHOICES_S)):
                            mask[flat2*len(DUR_CHOICES_S)+di] = 0
        obs = {"feats": feats, "mask": mask}
        return obs

    # ============== Baseline (legacy) ==============
    def baseline_action(self) -> Optional[Tuple[int,int]]:
        def delay(u, ci):
            return self.queues[u][ci].head_delay_ms(self.t_ms)
        # Rule 1: VO strict priority
        vo_candidates = []
        for u in range(self.n_users):
            if self.queues[u][CLASS_IDX["VO"]].pkts:
                vo_candidates.append((delay(u, CLASS_IDX["VO"]), u))
        if vo_candidates:
            u = max(vo_candidates)[1]
            dur_idx = len(DUR_CHOICES_S)-1
            return (u*3 + CLASS_IDX["VO"], dur_idx)
        # Rule 2: VI with delay>20ms strict priority
        vi_over = []
        for u in range(self.n_users):
            d = delay(u, CLASS_IDX["VI"])
            if self.queues[u][CLASS_IDX["VI"]].pkts and d > 20.0:
                vi_over.append((d, u))
        if vi_over:
            u = max(vi_over)[1]
            return (u*3 + CLASS_IDX["VI"], len(DUR_CHOICES_S)-1)
        # Rule 3: BE guarantee
        be_need = []
        for u in range(self.n_users):
            q = self.queues[u][CLASS_IDX["BE"]]
            if q.pkts and (self.t_ms - self.be_last_served[u] >= BE_GUARANTEE_MS):
                be_need.append((delay(u, CLASS_IDX["BE"]), u))
        if be_need:
            u = max(be_need)[1]
            return (u*3 + CLASS_IDX["BE"], 0)
        # Rule 4: longest VI/BE
        best = None
        for u in range(self.n_users):
            for ci in [CLASS_IDX["VI"], CLASS_IDX["BE"]]:
                bytes_q = self.queues[u][ci].total_bytes()
                if bytes_q > 0:
                    cand = (bytes_q, u, ci)
                    if (best is None) or (cand[0] > best[0]):
                        best = cand
        if best:
            _, u, ci = best
            return (u*3 + ci, len(DUR_CHOICES_S)-1)
        return None

# =========================
# PPO with action masking
# =========================
class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            logits = logits + (mask + 1e-8).log()
        self.dist = torch.distributions.Categorical(logits=logits)
    def sample(self):
        return self.dist.sample()
    def log_prob(self, a):
        return self.dist.log_prob(a)
    def entropy(self):
        return self.dist.entropy()

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.v(h)

@dataclass
class RolloutBuf:
    obs: List[np.ndarray] = field(default_factory=list)
    mask: List[np.ndarray] = field(default_factory=list)
    acts: List[int] = field(default_factory=list)
    rews: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    vals: List[float] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)

    def extend(self, other: 'RolloutBuf'):
        self.obs.extend(other.obs)
        self.mask.extend(other.mask)
        self.acts.extend(other.acts)
        self.rews.extend(other.rews)
        self.dones.extend(other.dones)
        self.vals.extend(other.vals)
        self.logps.extend(other.logps)

    def to_tensors(self, device):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.mask), dtype=torch.float32, device=device),
            torch.tensor(self.acts, dtype=torch.long, device=device),
            torch.tensor(self.rews, dtype=torch.float32, device=device),
            torch.tensor(self.dones, dtype=torch.float32, device=device),
            torch.tensor(self.vals, dtype=torch.float32, device=device),
            torch.tensor(self.logps, dtype=torch.float32, device=device),
        )

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5, device="cpu"):
        self.net = PolicyNet(obs_dim, act_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device

    def select(self, obs: np.ndarray, mask: np.ndarray):
        x = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)
        m = torch.tensor(mask[None, :], dtype=torch.float32, device=self.device)
        logits, v = self.net(x)
        dist = MaskedCategorical(logits.squeeze(0), mask=m.squeeze(0))
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(v.item()), float(logp.item())

    def update(self, buf: RolloutBuf, epochs=4, batch_size=4096):
        obs, mask, acts, rews, dones, vals, logps = buf.to_tensors(self.device)
        # GAE-Lambda
        adv = torch.zeros_like(rews)
        lastgaelam = 0
        for t in reversed(range(len(rews))):
            nextnonterminal = 1.0 - (dones[t] if t+1 < len(dones) else 0.0)
            nextvalue = vals[t+1] if t+1 < len(vals) else 0.0
            delta = rews[t] + self.gamma * nextvalue * nextnonterminal - vals[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.size(0)
        idx = torch.randperm(n)
        for _ in range(epochs):
            for start in range(0, n, batch_size):
                end = min(n, start+batch_size)
                b = idx[start:end]
                logits, v = self.net(obs[b])
                dist = MaskedCategorical(logits, mask=mask[b])
                new_logp = dist.log_prob(acts[b])
                entropy = dist.entropy().mean()

                ratio = (new_logp - logps[b]).exp()
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1.0-self.clip, 1.0+self.clip) * adv[b]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(v.squeeze(-1), ret[b])
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

# =========================
# Multiprocess rollout collection
# =========================

def worker_rollout(proc_id: int, env_kwargs: dict, policy_state: dict, steps: int, seed_offset: int, conn):
    try:
        # each worker builds its own env & agent
        # Create a copy of env_kwargs and update the seed to avoid duplicate keyword argument
        worker_env_kwargs = env_kwargs.copy()
        worker_env_kwargs['seed'] = env_kwargs.get('seed', 1) + seed_offset + proc_id
        env = SchedulerEnv(**worker_env_kwargs)
        obs_dim = env._observe()["feats"].shape[0]
        act_dim = env._observe()["mask"].shape[0]
        agent = PPOAgent(obs_dim, act_dim, device='cpu')
        agent.net.load_state_dict(policy_state)
        obs = env.reset()
        buf = RolloutBuf()
        done = False
        s = 0
        while not done and s < steps:
            feats = obs["feats"]
            mask = obs["mask"]
            # If all-invalid, idle
            if mask.max() <= 0:
                act = None
                a = 0
                v = 0.0
                logp = 0.0
            else:
                a, v, logp = agent.select(feats, mask)
                flat_idx = a // len(DUR_CHOICES_S)
                dur_idx = a % len(DUR_CHOICES_S)
                act = (flat_idx, dur_idx)
            next_obs, r, done, _ = env.step(act)

            buf.obs.append(feats)
            buf.mask.append(mask)
            buf.acts.append(a)
            buf.rews.append(r)
            buf.dones.append(float(done))
            buf.vals.append(v)
            buf.logps.append(logp)

            obs = next_obs
            s += 1
        conn.send(buf)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


def collect_rollout_mp(env_kwargs: dict, agent: PPOAgent, rollout_steps: int, procs: int) -> RolloutBuf:
    # split steps across workers
    steps_per = math.ceil(rollout_steps / procs)
    ctx = mp.get_context('spawn')
    children = []
    parent_conns = []
    policy_state = agent.net.state_dict()
    for i in range(procs):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=worker_rollout, args=(i, env_kwargs, policy_state, steps_per, 1000, child_conn))
        p.start()
        children.append(p)
        parent_conns.append(parent_conn)
    # gather
    buf = RolloutBuf()
    for conn in parent_conns:
        result = conn.recv()
        if isinstance(result, Exception):
            raise result
        buf.extend(result)
    for p in children:
        p.join()
    return buf

# =========================
# Training / Evaluation loops
# =========================

def build_env_kwargs(args) -> dict:
    traffic = {
        "VO": {"type": args.vo_type, "lambda_per_ms": args.vo_lambda, "bytes_per_pkt": args.vo_bytes, "period_ms": args.vo_period},
        "VI": {"type": args.vi_type, "lambda_per_ms": args.vi_lambda, "bytes_per_pkt": args.vi_bytes, "period_ms": args.vi_period},
        "BE": {"type": args.be_type, "lambda_per_ms": args.be_lambda, "bytes_per_pkt": args.be_bytes, "period_ms": args.be_period},
    }
    for k in ["VO","VI","BE"]:
        if traffic[k]["type"] == "poisson":
            traffic[k].pop("period_ms", None)
        else:
            traffic[k].pop("lambda_per_ms", None)
    env_kwargs = dict(
        n_users=args.users,
        minutes=args.minutes,
        seed=args.seed,
        traffic_conf=traffic,
        csv_window_ms=args.csv_window_ms if args.csv_out else None,
        csv_path=args.csv_out,
    )
    return env_kwargs


def run_baseline(env_kwargs: dict):
    env = SchedulerEnv(**env_kwargs)
    env.reset()
    done = False
    while not done:
        act = env.baseline_action()
        _, _, done, _ = env.step(act)
    report(env, label="Baseline")


def run_eval(env_kwargs: dict, agent: PPOAgent, csv_out: Optional[str] = None, csv_window_ms: Optional[int] = None):
    env = SchedulerEnv(**env_kwargs)
    if csv_out:
        env.enable_csv(csv_window_ms or 100, csv_out)
    obs = env.reset()
    done = False
    while not done:
        feats = obs["feats"]
        mask = obs["mask"]
        if mask.max() <= 0:
            act = None
        else:
            a, v, logp = agent.select(feats, mask)
            flat_idx = a // len(DUR_CHOICES_S)
            dur_idx = a % len(DUR_CHOICES_S)
            act = (flat_idx, dur_idx)
        obs, _, done, _ = env.step(act)
    report(env, label="RL Eval")


def run_train(env_kwargs: dict, iters=50, rollout_steps=4096, epochs=4, checkpoint_path="ppo_checkpoint.pt", device="cpu", procs: int = 1):
    # single env to size networks
    tmp_env = SchedulerEnv(**env_kwargs)
    obs_dim = tmp_env._observe()["feats"].shape[0]
    act_dim = tmp_env._observe()["mask"].shape[0]
    agent = PPOAgent(obs_dim, act_dim, device=device)

    for it in range(1, iters+1):
        if procs > 1:
            buf = collect_rollout_mp(env_kwargs, agent, rollout_steps, procs)
        else:
            # fallback single-process rollout
            env = SchedulerEnv(**env_kwargs)
            obs = env.reset()
            buf = RolloutBuf()
            done = False
            steps = 0
            while not done and steps < rollout_steps:
                feats = obs["feats"]
                mask = obs["mask"]
                a, v, logp = agent.select(feats, mask)
                flat_idx = a // len(DUR_CHOICES_S)
                dur_idx = a % len(DUR_CHOICES_S)
                act = (flat_idx, dur_idx)
                next_obs, r, done, _ = env.step(act)
                buf.obs.append(feats)
                buf.mask.append(mask)
                buf.acts.append(a)
                buf.rews.append(r)
                buf.dones.append(float(done))
                buf.vals.append(v)
                buf.logps.append(logp)
                obs = next_obs
                steps += 1
        agent.update(buf, epochs=epochs)
        if it % 10 == 0:
            torch.save(agent.net.state_dict(), checkpoint_path)
        print(f"[Iter {it}] rollout={len(buf.rews)}")
    torch.save(agent.net.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return agent


def load_agent(env_kwargs: dict, checkpoint_path: str, device="cpu") -> PPOAgent:
    tmp_env = SchedulerEnv(**env_kwargs)
    obs_dim = tmp_env._observe()["feats"].shape[0]
    act_dim = tmp_env._observe()["mask"].shape[0]
    agent = PPOAgent(obs_dim, act_dim, device=device)
    state = torch.load(checkpoint_path, map_location=device)
    agent.net.load_state_dict(state)
    return agent


def report(env: SchedulerEnv, label: str = ""):
    duration_s = env.steps/1000.0
    thr_mbps = env.bits_sent / duration_s / 1e6
    def p95(xs):
        return float(np.percentile(xs, 95)) if xs else 0.0
    print("========== Report:", label, "==========")
    print(f"Sim time: {duration_s:.3f}s  bits sent: {env.bits_sent:.0f}  Throughput: {thr_mbps:.2f} Mbps")
    print(f"Negative events: VO drops={env.neg_events_vo}  VI>50ms marks={env.neg_events_vi}")
    for c in CLASSES:
        print(f"p95 latency {c}: {p95(env.latency_samples[c]):.2f} ms  (samples={len(env.latency_samples[c])})")
    all_samples = env.latency_samples['VO'] + env.latency_samples['VI'] + env.latency_samples['BE']
    print(f"p95 latency overall: {p95(all_samples):.2f} ms  (samples={len(all_samples)})")
    print(f"Idle steps: {env.idle_steps}")

# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--users', type=int, default=16)
    p.add_argument('--minutes', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mode', choices=['baseline','train','eval'], default='baseline')
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--rollout', type=int, default=4096)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--checkpoint', type=str, default='ppo_checkpoint.pt')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--procs', type=int, default=1, help='number of processes for parallel rollout')

    # Traffic knobs
    p.add_argument('--vo_type', choices=['poisson','periodic'], default='poisson')
    p.add_argument('--vo_lambda', type=float, default=0.05)
    p.add_argument('--vo_bytes', type=int, default=400)
    p.add_argument('--vo_period', type=int, default=5)

    p.add_argument('--vi_type', choices=['poisson','periodic'], default='poisson')
    p.add_argument('--vi_lambda', type=float, default=0.03)
    p.add_argument('--vi_bytes', type=int, default=1200)
    p.add_argument('--vi_period', type=int, default=10)

    p.add_argument('--be_type', choices=['poisson','periodic'], default='periodic')
    p.add_argument('--be_lambda', type=float, default=0.01)
    p.add_argument('--be_bytes', type=int, default=1500)
    p.add_argument('--be_period', type=int, default=10)

    # CSV logging
    p.add_argument('--csv_out', type=str, default=None, help='CSV path for baseline or single-env runs')
    p.add_argument('--eval_csv_out', type=str, default=None, help='CSV path for eval after training')
    p.add_argument('--csv_window_ms', type=int, default=100)

    return p.parse_args()


def main():
    args = parse_args()
    env_kwargs = build_env_kwargs(args)

    if args.mode == 'baseline':
        run_baseline(env_kwargs)
    elif args.mode == 'train':
        agent = run_train(env_kwargs, iters=args.iters, rollout_steps=args.rollout, epochs=args.epochs,
                          checkpoint_path=args.checkpoint, device=args.device, procs=args.procs)
        print("Evaluating trained agent...")
        run_eval(env_kwargs, agent, csv_out=args.eval_csv_out or args.csv_out, csv_window_ms=args.csv_window_ms)
    else:  # eval
        agent = load_agent(env_kwargs, args.checkpoint, device=args.device)
        run_eval(env_kwargs, agent, csv_out=args.eval_csv_out or args.csv_out, csv_window_ms=args.csv_window_ms)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
