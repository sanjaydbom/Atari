from collections import deque
import random
from model import Model
from config import ACTION_SPACE,EPSILON,MIN_EPSILON,EPSILON_REDUCTION

class Agent():
    def __init__(self,action_space = ACTION_SPACE, epsilon = EPSILON, min_epsilon = MIN_EPSILON, epsilon_reduction = EPSILON_REDUCTION):
        self.model = Model()
        self.action_space = action_space

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_reduction = epsilon_reduction

