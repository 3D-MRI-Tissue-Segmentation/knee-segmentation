from util import Epsilon

from collections import defaultdict
import datetime
import gym
import numpy as np
import tensorflow.summary as tf_summary


class Q_Learn:

    def __init__(self, env,
                 epsilon, gamma=0.99, alpha=0.5,
                 obs_precision=1,
                 random_actions=False, verbose=False,
                 visualise=False, tf_writer=None,
                 reward_style=None):
        assert isinstance(env.action_space, gym.spaces.Discrete), "requires discrete actions space"
        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space required to be a box"
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.obs_space = env.observation_space
        assert type(self.obs_precision) is int
        self.obs_precision = [obs_precision] * env.observation_space.sample()
        assert len(env.observation_space.sample()) == len(obs_precision), "Size of obs space sample must equal precision list"

        self.actions = env.action_space
        self.Q = defaultdict(float)

        self.rewards = []
        self.ob = self.env.reset()
        self.reward = 0.0

        self.random_actions = random_actions
        self.verbose = verbose
        self.visualise = visualise
        self.tf_writer = tf_writer
        assert reward_style in [None, 'cumulative', 'punish', 'time']
        self.reward_style = reward_style

        self.steps = 0
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

    def modify_reward(self, r, done):
        reward_in = r
        if self.reward_style == 'punish':
            reward_in = r if not done else -20
        elif self.reward_style == 'cumulative':
            reward_in = self.reward
        return reward_in

    def step(self):
        self.steps += 1
        ob_str = self.box_to_string(self.ob)
        a = self.act(ob_str)
        ob_next, r, done, _ = self.env.step(a)
        if self.visualise:
            self.env.render()
        ob_next_str = self.box_to_string(ob_next)
        reward_in = self.modify_reward(r, done)
        self.update_Q(ob_str, a, reward_in, ob_next_str, done)
        self.reward += r
        if done:
            self.episodes += 1
            if self.tf_writer:
                with self.tf_writer.as_default():
                    tf_summary.scalar('episode reward', self.reward, step=self.episodes)
            if self.verbose:
                print(f"steps: {self.steps} - episode: {self.episodes} - r: {self.reward} - epsilon: {self.epsilon.value: .3f}")
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

def main(env_name="CartPole-v0", n_steps=1000000):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf_summary.create_file_writer(log_dir)

    env_ = gym.make(env_name)
    e = Epsilon(0.1, 0.9, 0.99)
    agent = Q_Learn(env_, e, obs_precision=1,
                    tf_writer=summary_writer)
    for i in range(n_steps):
        agent.step()

if __name__ == "__main__":
    for i in range(1):
        main(None)
