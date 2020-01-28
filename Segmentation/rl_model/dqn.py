from util import Epsilon, Uniform_Memory

import gym
import tensorflow as tf
import numpy as np


def mlp_make_network(obs_shape, n_actions, fcs):
    assert type(fcs) is list and len(fcs) >= 1, "fcs needs to be of type list"
    network_input = tf.keras.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(fcs[0], activation='relu',
                              kernel_initializer=tf.keras.initializers.he_normal())(network_input)
    for fc_size in fcs[1:]:
        x = tf.keras.layers.Dense(fc_size, activation='relu',
                                  kernel_initializer=tf.keras.initializers.he_normal())(x)
    network_output = tf.keras.layers.Dense(n_actions)(x)
    return tf.keras.Model(network_input, network_output)


class DQN_Agent:

    def __init__(self, env,
                 epsilon, gamma, alpha,
                 batch_size, lr, memory,
                 make_network, *make_network_args, random_actions=False):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.memory = memory
        self.network = make_network(self.obs_shape, self.n_actions, *make_network_args)
        self.network.compile(optimizer=tf.keras.optimizers.Adam(lr),
                             loss='mse')
        self.rewards = []
        self.ob = self.env.reset()
        self.reward = 0.0
        self.random_actions = random_actions

    def update_Q_network(self, ob, a, r, ob_next, done):
        state_qs = self.network.predict(ob)
        state_qs_next = self.network.predict(ob_next)
        max_q_next = state_qs_next.max(axis=1)
        for idx in range(self.batch_size):
            state_qs[idx, a[idx]] += self.alpha * (r[idx] + self.gamma * max_q_next[idx] * (1 - done[idx] - state_qs[idx, a[idx]]))
        self.network.fit(ob, state_qs, epochs=1, verbose=0)

    def act(self, ob):
        """ given current observation, pick the action with highest q value"""
        if self.random_actions:
            return self.env.action_space.sample()
        if np.random.random() < self.epsilon.value:
            return self.env.action_space.sample()
        ob = np.asarray([ob], dtype="float32")
        states_qs = self.network.predict(ob)[0]
        max_q = max(states_qs)  # Gets max q value
        actions_with_max_q = [a for a, q in enumerate(states_qs) if q == max_q]  # List of actions with max q
        return np.random.choice(actions_with_max_q)  # In the case multiple actions have the max q value

    def train(self):
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
        obs, actions, rewards, next_obs, dones = self.memory.batch_to_np(batch)
        self.update_Q_network(obs, actions, rewards, next_obs, dones)

    def step(self):
        a = self.act(self.ob)
        ob_next, r, done, _ = self.env.step(a)
        self.memory.remember(self.ob, a, r, ob_next, done)
        self.memory
        self.reward += r
        if done:
            self.rewards.append(self.reward)
            self.train()
            self.reward = 0.0
            self.ob = self.env.reset()
        else:
            self.ob = ob_next


if __name__ == "__main__":
    print("running")
    env = gym.make("CartPole-v0")
    e = Epsilon(0.01, 0.9, 0.99)
    gamma = 0.99
    alpha = 0.5
    batch_size = 120
    lr = 0.01
    memory = Uniform_Memory(10000)
    mlp_make_network_args = [[5, 10]]

    agent = DQN_Agent(env,
                      e, gamma, alpha,
                      batch_size, lr, memory,
                      mlp_make_network, *mlp_make_network_args,
                      random_actions=False)

    n_steps = 50000
    percent = n_steps * 0.1
    for i in range(n_steps):
        if i % percent == 0:
            print(i)
        agent.step()
    env.env.close()

    import matplotlib.pyplot as plt
    plt.plot(agent.rewards)
    N = 100
    plt.plot(np.convolve(agent.rewards, np.ones((N,)) / N, mode='valid'))
    plt.show()
