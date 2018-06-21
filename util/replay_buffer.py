# replay_buffer.py
# Reference-to: https://github.com/chainer/chainerrl/blob/master/chainerrl/replay_buffer.py

from collections import deque
import numpy as np
from chainerrl.replay_buffer import AbstractReplayBuffer


class SimpleReplayBuffer(AbstractReplayBuffer):
    """最も単純な ReplayBuffer"""

    def __init__(self, capacity=10**6):
        self.memory = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state=None, next_action=None, is_state_terminal=False):
        experience = dict(state=state, action=action, reward=reward,
                          next_state=next_state, next_action=next_action,
                          is_state_terminal=is_state_terminal)
        self.memory.append(experience)
        # Point: ↑ deque は、append() 時にサイズが maxlen を超えると先頭から削除される仕組み。

    def sample(self, n):
        assert len(self.memory) >= n
        return list(np.random.choice(self.memory, n, replace=False))

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def stop_current_episode(self):
        pass


class BestReplayBuffer(SimpleReplayBuffer):
    """報酬の高いものを優先的に記憶する ReplayBuffer"""

    def __init__(self, capacity=1000):
        super().__init__(capacity)

    # TODO: より報酬の高いものを優先的に残すような実装をする
    # def append(self, state, action, reward, next_state=None, next_action=None, is_state_terminal=False):
    #     pass
