# explorers.py
# # original: https://github.com/chainer/chainerrl/blob/master/chainerrl/explorers/epsilon_greedy.py

import numpy as np

from chainerrl import explorer


def select_action_epsilon_greedily(epsilon, random_action_func,
                                   greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func()
    else:
        return greedy_action_func()


class ConstantEpsilonGreedy(explorer.Explorer):
    """ε-greedy 法（ε固定）"""

    def __init__(self, epsilon, random_action_func):
        assert epsilon >= 0 and epsilon <= 1
        self.epsilon = epsilon
        self.random_action_func = random_action_func

    def select_action(self, t, greedy_action_func, action_value=None):
        return select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func)

    def __repr__(self):
        return 'ConstantEpsilonGreedy(epsilon={})'.format(self.epsilon)


class LinearDecayEpsilonGreedy(explorer.Explorer):
    """ε-greedy 法（ε線型減衰）"""

    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.epsilon = start_epsilon

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        return select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func)

    def __repr__(self):
        return 'LinearDecayEpsilonGreedy(epsilon={})'.format(self.epsilon)
