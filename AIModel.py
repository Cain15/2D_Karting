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


class RandomAgent:
    def __init__(self):
        pass
    def act(self, state):
        x = random.randint(0,2)
        y = random.randint(2,4)
        return x,y
    def update(self, state, action, reward, next_state):
        pass

class QAgent:
    def __init__(self):
        self.weights = np.zeros(23)
        self.actions = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4)]
        self.alpha = 0.005
        self.gamma = 0.99
        self.epsilon = 1.0
        self.start_time = time.time()
        self.eps_decay = 0.99999
        self.epsilon_min = 0.05

    def get_features(self, state, action):
        """
        :param state: List of state features
        :param action: (a1, a2), a1: steering, a2: acceleration
        :return: features
        """
        actions = [0] * 9
        actions[self.actions.index(action)] = 1
        features = np.array(state + actions + [1.0])
        return features

    def q_value(self, state, action):
        f = self.get_features(state, action)
        return np.dot(self.weights, f)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.q_value(state, action) for action in self.actions]
        return self.actions[np.argmax(q_values)]


    def update(self, state, action, reward, next_state):
        max_a = max([self.q_value(next_state, action) for action in self.actions])
        diff = reward + self.gamma * max_a - self.q_value(state, action)
        diff = np.clip(diff, -1.0, 1.0)
        f = self.get_features(state, action)
        self.weights += self.alpha * diff * f
        if (time.time() - self.start_time) > 60:
            self.start_time = time.time()
            with open("weights.txt", "a") as file:
                file.write(str(self.weights))
                file.write("\n")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay)








