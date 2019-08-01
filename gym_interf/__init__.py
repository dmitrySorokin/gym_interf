from gym.envs.registration import register
from .envs import InterfEnv

register(id='interf-v1',
         entry_point='gym_interf.envs:InterfEnv')
