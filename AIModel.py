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


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.steer_head = nn.Linear(128, 3)
        self.throttle_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        steer_logits = self.steer_head(features)
        throttle_logits = self.throttle_head(features)
        value = self.value_head(features)
        return steer_logits, throttle_logits, value

class PPOAgent:
    def __init__(self, state_dim=8, action_dim=9):

        self.steer_map = {0: 0, 1: 1, 2: 2}  # left, right, neutral
        self.throttle_map = {0: 3, 1: 2, 2: 4}  # decelerate, neutral, accelerate

        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.lr = 3e-4
        self.epochs = 10
        self.batch_size = 256

        self.net = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.memory = []

        self.last_save_time = time.time()
        self.save_interval = 300

        self.rollout_size = 2048

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            steer_logits, throttle_logits, value = self.net(state)

            steer_dist = torch.distributions.Categorical(logits=steer_logits)
            throttle_dist = torch.distributions.Categorical(logits=throttle_logits)

            steer_action = steer_dist.sample()
            throttle_action = throttle_dist.sample()

        return (
            (self.steer_map[steer_action.item()], self.throttle_map[throttle_action.item()]),
            steer_action.item(),
            throttle_action.item(),
            steer_dist.log_prob(steer_action),
            throttle_dist.log_prob(throttle_action),
            value.item()
        )

    def store(self, transition):
        self.memory.append(transition)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1-dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1-dones[i]) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self):

        states, steer_actions, throttle_actions, \
            old_steer_log_probs, old_throttle_log_probs, \
            rewards, dones, values = zip(*self.memory)
        # Bootstrap value for last state if rollout didn't end episode
        if dones[-1]:
            next_value = 0
        else:
            last_state = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, next_val_tensor = self.net(last_state)
                next_value = next_val_tensor.item()

        advantages = self.compute_gae(rewards, list(values), dones, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]

        states = torch.tensor(states, dtype=torch.float32)
        steer_actions = torch.tensor(steer_actions)
        throttle_actions = torch.tensor(throttle_actions)

        old_steer_log_probs = torch.stack(old_steer_log_probs).detach()
        old_throttle_log_probs = torch.stack(old_throttle_log_probs).detach()

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            steer_logits, throttle_logits, values_pred = self.net(states)

            steer_dist = torch.distributions.Categorical(logits=steer_logits)
            throttle_dist = torch.distributions.Categorical(logits=throttle_logits)

            new_steer_log_probs = steer_dist.log_prob(steer_actions)
            new_throttle_log_probs = throttle_dist.log_prob(throttle_actions)

            ratio_steer = (new_steer_log_probs - old_steer_log_probs).exp()
            ratio_throttle = (new_throttle_log_probs - old_throttle_log_probs).exp()

            ratio = ratio_steer * ratio_throttle

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values_pred.squeeze()).pow(2).mean()

            entropy = steer_dist.entropy().mean() + throttle_dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

        self.memory = []

        if time.time() - self.last_save_time > self.save_interval:
            self.last_save_time = time.time()
            self.save()

    def save(self, filename="ppo_model.pt"):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'lam': self.lam,
            'clip': self.clip,
            'lr': self.lr,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }, filename)

        print(f"PPO model saved to {filename}")

    def load(self, filename="ppo_model.pt"):
        checkpoint = torch.load(filename)

        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Optional: restore hyperparameters
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.lam = checkpoint.get('lam', self.lam)
        self.clip = checkpoint.get('clip', self.clip)
        self.lr = checkpoint.get('lr', self.lr)
        self.epochs = checkpoint.get('epochs', self.epochs)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)

        print(f"PPO model loaded from {filename}")








