from collections import deque
import random
import numpy as np

from .config import MEMORY_SIZE
class PriorityReplay():
    """Priority Replay Buffer to Sample Memories proportional to TD error
    
    Implements a Priority Replay Buffer to store memories in order of TD error and uses proportional sampling to sample experiences.
    The underlying data structure to store the memories is a Sum Tree.  See https://arxiv.org/pdf/1511.05952 for more info

    Args:
        Capacity (int): number of experiences stored-default 300000

    Attrubutes:
        tree (list): a array implementation of the Sum Tree where the child of node i are 2i+1 and 2i+2
        data (list): a array of the leaf nodes with the values of the memories
        size (int): the number of memories stored
        data_pointer(int): the index of where to store the next memories. Loops around once all memories slots are full
    """
    def __init__(self, capacity:int = MEMORY_SIZE):
        self.capacity = capacity
        self.tree = [0] * (2 * self.capacity - 1)
        self.data = [None] * capacity
        self.size = 0
        self.data_pointer = 0
        self.alpha = 0.5
        self.beta = 0.4
        self.epsilon = 0.01

    def append(self, element:list) -> None:
        """Stores memories and updates the Sum Tree

        args:
            priority(float): TD error of the element
            element(list): a list of (state, action, reward, next state)
        """
        priority = max(self.tree[-self.capacity:]) if self.size > 0 else 1.0
        self.data[self.data_pointer] = element
        idx = self.data_pointer + self.capacity - 1
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update_td(self, idx, new_td):
        new_td = abs(new_td + self.epsilon) ** self.alpha
        delta = new_td - self.tree[idx]
        self.tree[idx] = new_td
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def update_all_td(self, indicies, td_errors):
        for idx, new_td in zip(indicies, td_errors):
            self.update_td(idx, new_td)

    def get_total(self):
        """
        Returns the sum of the total TD errors
        """
        return self.tree[0]
    
    def sample(self, num_samples:int = 32):
        """
        randomly samples num_samples memories from the sum tree
        """
        if self.size == 0:
            return None
        samples = [0] * num_samples
        indicies = [0] * num_samples
        weights = [0] * num_samples
        segment = self.get_total() / num_samples
        for i in range(num_samples):
            s = random.uniform(segment * i,segment * (i+1))
            idx = 0
            while idx < self.capacity - 1:
                if s <= self.tree[2 * idx + 1]:
                    idx = 2 * idx + 1
                else:
                    s -= self.tree[2 * idx + 1]
                    idx = 2 * idx + 2
            samples[i] = self.data[idx - self.capacity + 1]
            indicies[i] = idx
            weights[i] = (self.size * self.tree[idx]/ self.get_total() )** -self.beta
        return samples, indicies, weights
    
    def __len__(self):
        return self.size
    '''
    def save_memory(self):
        numpy_tree = np.array(self.tree)
        numpy_data = np.array(self.data)
        np.savez_compressed("agent_weights/numpy_tree.npz", data = numpy_tree)
        np.savez_compressed("agent_weights/numpy_data.npz", data = numpy_data)

    def get_data(self):
        return self.size, self.data_pointer
    
    '''
    def anneal_beta(self):
        self.beta += 0.6*(1e-7)