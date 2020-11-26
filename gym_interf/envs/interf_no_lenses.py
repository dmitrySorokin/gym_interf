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
