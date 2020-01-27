from util import Epsilon

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


class Memoryless_DQN_Agent:

    def __init__(self, env,
                 epsilon, gamma, alpha,
                 batch_size, lr,
                 make_network, *make_network_args):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size

        self.network = make_network(self.obs_shape, self.n_actions, *make_network_args)
        self.network.compile(optimizer=tf.keras.optimizers.Adam(lr),
                             loss='mse')

        self.rewards = []
        self.ob = self.env.reset()
        self.reward = 0.0

    def update_Q_network(self, ob, r, a, ob_next, done):
        ob = np.asarray([ob], dtype="float32")
        state_qs = self.network.predict(ob)[0]
        ob_next = np.asarray([ob_next], dtype="float32")
        state_qs_next = self.network.predict([ob_next])[0]
        max_q_next = max([state_qs_next[a] for a in range(self.n_actions)])
        state_qs[a] += self.alpha * (r + self.gamma * max_q_next * (1 - done) - state_qs[a])
        state_qs = np.asarray([state_qs], dtype="float32")
        self.network.fit(ob, state_qs, epochs=1, verbose=0)

    def act(self, ob):
        """ given current observation, pick the action with highest q value"""
        if np.random.random() < self.epsilon.value:
            return self.env.action_space.sample()
        ob = np.asarray([ob], dtype="float32")
        states_qs = self.network.predict(ob)[0]
        max_q = max(states_qs)  # Gets max q value
        actions_with_max_q = [a for a, q in enumerate(states_qs) if q == max_q]  # List of actions with max q
        return np.random.choice(actions_with_max_q)  # In the case multiple actions have the max q value

    def step(self):
        a = self.act(self.ob)
        ob_next, r, done, _ = self.env.step(a)
        self.update_Q_network(self.ob, r, a, ob_next, done)
        self.reward += r
        if done:
            self.rewards.append(self.reward)
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
    batch_size = 32
    lr = 0.001
    mlp_make_network_args = [[5, 10]]

    dqn = Memoryless_DQN_Agent(env,
                               e, gamma, alpha,
                               batch_size, lr,
                               mlp_make_network, *mlp_make_network_args)

    n_steps = 10000
    percent = n_steps * 0.01
    for i in range(n_steps):
        if i % percent == 0:
            print(i)
        dqn.step()
    env.env.close()

    import matplotlib.pyplot as plt
    plt.plot(dqn.rewards)
    plt.show()
