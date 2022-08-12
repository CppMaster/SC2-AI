import gym
import numpy as np
from gym.spaces import Discrete, Box


class AddActionAndRewardToObservationWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, reward_scale: float = 1.0):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Only Discrete observation space handled"
        assert isinstance(env.action_space, Discrete), "Only Discrete action space handled"
        self.observation_space = Box(-1.0, 3.0, (env.observation_space.shape[0] + env.action_space.n + 1,))
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        transformed_obs = np.concatenate([obs, np.identity(self.env.action_space.n)[action],
                                         np.array([reward * self.reward_scale])])
        return transformed_obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        transformed_obs = np.concatenate([obs, np.zeros((self.env.action_space.n,)), np.array([0])])
        return transformed_obs
