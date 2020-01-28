from collections import defaultdict
from util import Epsilon
import gym
import numpy as np


class Q_Learn:

    def __init__(self, env,
                 epsilon, gamma=0.99, alpha=0.5,
                 obs_precision=[1, 1, 1, 1],
                 random_actions=False, verbose=False):
        assert isinstance(env.action_space, gym.spaces.Discrete), "requires discrete actions space"
        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space required to be a box"
        assert len(env.observation_space.sample()) == len(obs_precision), "Size of obs space sample must equal precision list"
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.obs_space = env.observation_space
        self.obs_precision = obs_precision

        self.actions = env.action_space
        self.Q = defaultdict(float)

        self.rewards = []
        self.ob = self.env.reset()
        self.reward = 0.0

        self.random_actions = random_actions
        self.verbose = verbose
        self.episodes = 0

    def update_Q(self, ob_str, a, r, ob_next_str, done):
        max_q_next = max([self.Q[ob_next_str, a] for a in range(self.actions.n)])
        if not done:
            self.Q[ob_str, a] += self.alpha * (r + self.gamma * max_q_next - self.Q[ob_str, a])

    def act(self, ob_str):
        """ given current observation, pick the action with highest q value"""
        if self.random_actions:
            return self.env.action_space.sample()
        if np.random.random() < self.epsilon.value:
            return self.actions.sample()
        states_qs = {a: self.Q[ob_str, a] for a in range(self.actions.n)}
        max_q = max(states_qs.values())  # Gets max q value
        actions_with_max_q = [a for a, q in states_qs.items() if q == max_q]  # List of actions with max q
        return np.random.choice(actions_with_max_q)  # In the case multiple actions have the max q value

    def step(self, visualise=False):
        ob_str = self.box_to_string(self.ob)
        a = self.act(ob_str)
        ob_next, r, done, _ = self.env.step(a)
        if visualise:
            self.env.render()
        ob_next_str = self.box_to_string(ob_next)
        self.update_Q(ob_str, a, r, ob_next_str, done)
        self.reward += r
        if done:
            self.rewards.append(self.reward)
            self.epsilon.update_epsilon()
            self.reward = 0.0
            self.ob = self.env.reset()
        else:
            self.ob = ob_next

    def box_to_string(self, ob):
        ob_str = ""
        for idx, o in enumerate(ob):
            p = self.obs_precision[idx]
            ob_str += f"_{o:.{p}f}_"
        return ob_str

if __name__ == "__main__":
    print("Running...")

    env_ = gym.make("CartPole-v0")
    e = Epsilon(0.1, 0.9, 0.99)
    agent = Q_Learn(env_, e)
    n_steps = 1000000
    for i in range(n_steps):
        agent.step()
    env_.env.close()

    import matplotlib.pyplot as plt
    plt.plot(agent.rewards)
    N = 100
    plt.plot(np.convolve(agent.rewards, np.ones((N,)) / N, mode='valid'))
    plt.show()
