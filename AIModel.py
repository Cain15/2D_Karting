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

        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

class PPOAgent:
    def __init__(self, state_dim=8, action_dim=9):

        self.actions = [(0,2), (0,3), (0,4),
                        (1,2), (1,3), (1,4),
                        (2,2), (2,3), (2,4)]

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
            logits, value = self.net(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        return (
            self.actions[action.item()],
            action.item(),
            dist.log_prob(action),
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

        states, actions, log_probs, rewards, dones, values = zip(*self.memory)
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
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(log_probs).detach()
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):

            logits, values_pred = self.net(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values_pred.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()

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








