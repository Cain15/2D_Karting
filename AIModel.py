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
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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

class DDQNAgent:
    def __init__(self):
        self.actions = [(0,2), (0,3), (0,4),
                        (1,2), (1,3), (1,4),
                        (2,2), (2,3), (2,4)]

        self.state_dim = 14
        self.action_dim = 9

        self.q_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)

        self.buffer = ReplayBuffer()

        self.gamma = 0.99
        self.batch_size = 256

        self.last_save_time = time.time()
        self.save_interval = 300  # 5 minutes

        self.count = 0
        self.train_frequency = 10

        self.epsilon = 0.9
        self.epsilon_step = 0.005
        self.epsilon_min = 0.1

    def q_value(self, state):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.q_net(s).squeeze().numpy()

    def act(self, state):
        self.epsilon -= self.epsilon_step
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(s)
            action_idx = torch.argmax(q_values).item()

            return self.actions[action_idx]

    def update(self, state, action, reward, next_state, done=None):
        action_idx = self.actions.index(action)

        if done is None:
            done = False
            print("Something went wrong")

        self.buffer.push(state, action_idx, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        self.count += 1
        if self.count % self.train_frequency != 0:
            return
        self.count = 0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_vals = self.q_net(states)
        print(q_vals.mean().item(), q_vals.max().item())
        q_val = q_vals.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze()
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        tau = 0.005
        for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

        if time.time() - self.last_save_time > self.save_interval:
            self.last_save_time = time.time()
            self.save()

    def save(self, filename="ddqn_model.pth"):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print("Model saved")

    def load(self, filename="ddqn_model.pth"):
        checkpoint = torch.load(filename)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']













