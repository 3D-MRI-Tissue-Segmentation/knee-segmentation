import numpy as np
from collections import deque

class Epsilon():
    def __init__(self, low, high, decay):
        assert 0. < decay < 1.0
        assert 0. <= high <= 1.0
        assert 0. <= low < 1.0
        self.high = high
        self.low = low
        self.decay = decay
        self.value = high

    def update_epsilon(self):
        new_value = self.value * self.decay
        if new_value > self.low:
            self.value = new_value

class Uniform_Memory:
    def __init__(self, memory_length):
        self.samples = deque(maxlen=memory_length)

    def remember(self, state, action, reward, next_state):
        self.samples.append([state, action, reward, next_state])

    def sample(self, n_samples):
        if n_samples < len(self.sample):
            return
        return np.random.sample(self.sample, n_samples)

    def batch_to_np(self, batch):
        states = np.array([val[0] for val in batch])
        actions = np.array([val[1] for val in batch])
        rewards = np.array([val[2] for val in batch])
        next_states = np.array([val[3] for val in batch])
        return states, actions, rewards, next_states
