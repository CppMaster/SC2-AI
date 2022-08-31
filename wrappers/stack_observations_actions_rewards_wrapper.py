from dataclasses import dataclass
from typing import List, Optional

import gym
import numpy as np
from gym.spaces import Discrete, Box

from utils.value.value_stack import ValueStack


@dataclass
class ValueStackConfig:
    n_last_values: int = 1,
    n_average_prev_last_values: Optional[List[int]] = None


class StackObservationsActionRewardsWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, reward_scale: float = 1.0,
                 observation_value_stack_config=ValueStackConfig(1, [5]),
                 action_value_stack_config=ValueStackConfig(10, [20]),
                 reward_value_stack_config=ValueStackConfig(20, [50])
                 ):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Only Discrete observation space handled"
        assert isinstance(env.action_space, Discrete), "Only Discrete action space handled"
        self.observation_value_stack = ValueStack(
            env.observation_space.shape, observation_value_stack_config.n_last_values,
            observation_value_stack_config.n_average_prev_last_values)
        self.action_value_stack = ValueStack(
            (env.action_space.n,), action_value_stack_config.n_last_values,
            action_value_stack_config.n_average_prev_last_values)
        self.reward_value_stack = ValueStack(
            (1,), reward_value_stack_config.n_last_values, reward_value_stack_config.n_average_prev_last_values
        )
        self.observation_space = Box(-1.0, 3.0, (self.observation_value_stack.n_return_elements +
                                                 self.action_value_stack.n_return_elements +
                                                 self.reward_value_stack.n_return_elements,))
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.observation_value_stack.add_value(obs)
        self.action_value_stack.add_value(np.identity(self.env.action_space.n)[action])
        self.reward_value_stack.add_value(np.array([reward * self.reward_scale]))
        stacked_observation = self.get_stacked_observation()
        return stacked_observation, reward, done, info

    def reset(self, **kwargs):
        self.observation_value_stack.reset()
        self.action_value_stack.reset()
        self.reward_value_stack.reset()
        obs = self.env.reset(**kwargs)
        self.observation_value_stack.add_value(obs)
        return self.get_stacked_observation()

    def get_stacked_observation(self) -> np.ndarray:
        return np.concatenate([
            self.observation_value_stack.get_values().flatten(), self.action_value_stack.get_values().flatten(),
            self.reward_value_stack.get_values().flatten()])

