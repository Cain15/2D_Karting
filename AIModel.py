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


class Action(enum.Enum):
    left = 0
    right = 1
    neutral = 2
    accelerate = 3
    decelerate = 4


class RandomAgent:
    def __init__(self):
        pass
    def act(self):
        x = random.randint(0,2)
        y = random.randint(2,4)
        return Action(x), Action(y)


