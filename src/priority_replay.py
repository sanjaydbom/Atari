from collections import deque
import random

class PriorityReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * self.capacity)
        self.data = [None] * capacity
        self.size = 0
        self.data_pointer = 0
        self.epsilon = 1e-3

    def add(self, priority, element):
        self.data[self.data_pointer] = element
        idx = self.data_pointer + self.capacity -1
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_total(self):
        return self.tree[0]
    
    def sample(self):
        if self.size == 0:
            return None
        s = random.uniform(0,self.get_total())
        idx = 0
        while idx < self.capacity - 1:
            if s <= self.tree[2 * idx + 1]:
                idx = 2 * idx + 1
            else:
                s -= self.tree[2 * idx + 1]
                idx = 2 * idx + 2
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]
    
