"""
Rainbow DQN Implementation
Combines:
- Double DQN
- Dueling Networks
- NoisyNets (replaces epsilon-greedy)
- Prioritized Experience Replay (PER)
- N-step Returns

Actions:
    - (left, accelerate)
    - (right, accelerate)
    - (neutral, accelerate)
    - (left, decelerate)
    - (right, decelerate)
    - (neutral, decelerate)
    - (left, neutral)
    - (right, neutral)
    - (neutral, neutral)
"""
import enum
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np


class Action(enum.Enum):
    left = 0
    right = 1
    neutral = 2
    decelerate = 3
    accelerate = 4


# NoisyLinear for exploration
class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer."""

    def __init__(self, in_features: int, out_features: int, sigma: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.full((out_features, in_features),
                                               sigma / math.sqrt(in_features)))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.full((out_features,),
                                               sigma / math.sqrt(in_features)))

        self.register_buffer('eps_w', torch.zeros(out_features, in_features))
        self.register_buffer('eps_b', torch.zeros(out_features))

        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        p = self._f(torch.randn(self.in_features, device=self.mu_w.device))
        q = self._f(torch.randn(self.out_features, device=self.mu_w.device))
        self.eps_w.copy_(q.unsqueeze(1) * p.unsqueeze(0))
        self.eps_b.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.mu_w + self.sigma_w * self.eps_w
            b = self.mu_b + self.sigma_b * self.eps_b
        else:
            w, b = self.mu_w, self.mu_b
        return F.linear(x, w, b)


# Dueling + Noisy DQN
class RainbowDQN(nn.Module):
    """Dueling architecture with NoisyNet heads."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(256, 256),
            nn.ReLU(),
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        V = self.value(shared)
        A = self.advantage(shared)
        return V + (A - A.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


#  Prioritized Replay Buffer (PER)
# Use a SumTree for fast sampling
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 200_000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 2_000_000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.max_priority = 1.0
        self.epsilon = 1e-5

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start +
                   self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority ** self.alpha,
                      (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        self.frame += 1
        indices, priorities, batch = [], [], []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        total = self.tree.total
        n = self.tree.n_entries
        probs = np.array(priorities) / total
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()

        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
            indices,
            torch.tensor(weights, dtype=torch.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            priority = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries


#  N-step Return Buffer
class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def ready(self) -> bool:
        return len(self.buffer) >= self.n

    def get(self):
        """Return (s0, a0, n-step-return, s_n, done_n)."""
        state, action = self.buffer[0][0], self.buffer[0][1]
        n_return = 0.0
        for i, (_, _, r, ns, d) in enumerate(self.buffer):
            n_return += (self.gamma ** i) * r
            if d:
                return state, action, n_return, ns, True
        _, _, _, last_ns, last_done = self.buffer[-1]
        return state, action, n_return, last_ns, last_done

    def pop(self):
        self.buffer.popleft()

    def clear(self):
        self.buffer.clear()


#  Rainbow Agent
class DDQNAgent:
    def __init__(self):
        self.actions = [
            (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 2), (2, 3), (2, 4),
        ]
        self.state_dim = 18
        self.action_dim = 9

        self.q_net = RainbowDQN(self.state_dim, self.action_dim)
        self.target_net = RainbowDQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        # q_net stays in train() mode permanently so NoisyLinear always
        # applies noise during both inference and training.
        # target_net stays in eval() so it gives deterministic targets.
        self.q_net.train()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)

        self.buffer = PrioritizedReplayBuffer(
            capacity=200_000,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=2_000_000,
        )

        self.gamma = 0.99
        self.batch_size = 256
        self.n_steps = 10
        self._nstep_buffers: dict[int, NStepBuffer] = {}

        self.count = 0
        self.train_frequency = 60

        self.last_save_time = time.time()
        self.save_interval = 300

    def _get_nstep(self, player_id: int) -> NStepBuffer:
        if player_id not in self._nstep_buffers:
            self._nstep_buffers[player_id] = NStepBuffer(self.n_steps, self.gamma)
        return self._nstep_buffers[player_id]

    def act(self, state) -> tuple:
        """
        q_net is always in train() mode, so NoisyLinear noise is always
        active — no epsilon, no eval/train toggle needed.
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_idx = torch.argmax(self.q_net(s)).item()
            return self.actions[action_idx]

    def q_value(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.q_net(s).squeeze().numpy()

    def update(self, state, action, reward, next_state, done: bool = False,
               player_id: int = 0):
        action_idx = self.actions.index(action)
        nb = self._get_nstep(player_id)
        nb.push(state, action_idx, reward, next_state, done)

        if done:
            while nb.ready():
                s0, a0, ret, sn, dn = nb.get()
                self.buffer.push(s0, a0, ret, sn, dn)
                nb.pop()
            nb.clear()
        elif nb.ready():
            s0, a0, ret, sn, dn = nb.get()
            self.buffer.push(s0, a0, ret, sn, dn)
            nb.pop()

        if len(self.buffer) < self.batch_size:
            return

        self.count += 1
        if self.count % self.train_frequency != 0:
            return
        self.count = 0

        self._train_step()

        if time.time() - self.last_save_time > self.save_interval:
            self.last_save_time = time.time()
            self.save()

    def _train_step(self):
        (states, actions, rewards, next_states,
         dones, indices, weights) = self.buffer.sample(self.batch_size)

        q_vals = self.q_net(states)
        q_val = q_vals.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            # Double DQN: q_net picks action, target_net scores it
            best_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze()
            target = rewards + (1 - dones) * (self.gamma ** self.n_steps) * next_q

        td_errors = (target - q_val).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        loss = (weights * F.smooth_l1_loss(q_val, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        tau = 0.005
        for tp, qp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tau * qp.data + (1.0 - tau) * tp.data)

        self.q_net.reset_noise()
        self.target_net.reset_noise()

    def save(self, filename: str = "rainbow_model.pth"):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)
        print(f"Model saved → {filename}")

    def load(self, filename: str = "rainbow_model.pth"):
        checkpoint = torch.load(filename, weights_only=False)
        try:
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Model loaded ← {filename}")
        except RuntimeError:
            print("WARNING: checkpoint architecture mismatch — starting fresh.")
