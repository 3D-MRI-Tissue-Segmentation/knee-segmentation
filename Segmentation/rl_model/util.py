import numpy as np
from collections import deque

import random

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

    def remember(self, ob, action, reward, next_ob, done):
        self.samples.append([ob, action, reward, next_ob, done])

    def sample(self, n_samples):
        if n_samples > len(self.samples):
            return
        return random.sample(self.samples, n_samples)

    def batch_to_np(self, batch):
        obs = np.array([val[0] for val in batch], dtype="float32")
        actions = np.array([val[1] for val in batch], dtype="int8")
        rewards = np.array([val[2] for val in batch], dtype="float32")
        next_obs = np.array([val[3] for val in batch], dtype="float32")
        dones = np.array([val[4] for val in batch])
        return obs, actions, rewards, next_obs, dones

def plot_rewards(agent, N=100):
    import matplotlib.pyplot as plt
    plt.plot(agent.rewards)
    plt.plot(np.convolve(agent.rewards, np.ones((N,)) / N, mode='valid'))
    plt.show()
