from gym.envs.registration import register

register(id='interf-v1',
         entry_point='gym_interf.envs:InterfEnv')
