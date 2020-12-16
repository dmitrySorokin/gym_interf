import gym
import numpy as np

from .interf_env import InterfEnv


class InterfNoLenses(InterfEnv):
    n_actions = 4
    action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)

    def step(self, actions):
        return super().step([*actions, 0])

    def reset(self, actions=None):
        if actions is None:
            actions = InterfNoLenses.action_space.sample()
        return super().reset([*actions, 0])

    def get_keys_to_action(self):
        return {
            (ord('w'),): 0,
            (ord('s'),): 1,
            (ord('a'),): 2,
            (ord('d'),): 3,
            (ord('i'),): 4,
            (ord('k'),): 5,
            (ord('j'),): 6,
            (ord('l'),): 7,
        }
