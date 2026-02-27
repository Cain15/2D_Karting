"""
Actions:
    - (left, accelerate)
    - (right, accelerate))
    - (neutral, accelerate)
    - (left, decelerate)
    - (right, decelerate)
    - (neutral, decelerate)
    - (left, neutral)
    - (right, neutral)
    - (neutral, neutral)
"""
import enum
import random
import numpy as np
import time


class Action(enum.Enum):
    left = 0
    right = 1
    neutral = 2
    decelerate = 3
    accelerate = 4

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self):
        self.actions = [(0,2), (0,3), (0,4),
                        (1,2), (1,3), (1,4),
                        (2,2), (2,3), (2,4)]

        self.state_dim = 10
        self.action_dim = 9

        self.q_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0005)

        self.buffer = ReplayBuffer()

        self.gamma = 0.99
        self.batch_size = 128

        self.epsilon = 0.3
        self.step = 0.000001
        self.epsilon_min = 0.05

        self.update_counter = 0

        self.last_save_time = time.time()
        self.save_interval = 300  # 5 minutes

    def q_value(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.q_net(s).squeeze().numpy()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(s)
            action_idx = torch.argmax(q_values).item()

        return self.actions[action_idx]

    def update(self, state, action, reward, next_state, done = None):

        action_idx = self.actions.index(action)

        if not done:
            done = False

        self.buffer.push(state, action_idx, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_vals = self.q_net(states)
        q_val = q_vals.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze()
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1

        tau = 0.005  # Factor voor geleidelijke update
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

        self.epsilon = max(self.epsilon_min, self.epsilon - self.step)

        if time.time() - self.last_save_time > self.save_interval:
            self.last_save_time = time.time()
            self.save()
            print("Model saved")

    def save(self, filename="dqn_model.pt"):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename="dqn_model.pt"):
        checkpoint = torch.load(filename)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)









