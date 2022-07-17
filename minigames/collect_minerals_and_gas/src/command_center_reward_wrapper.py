from typing import Optional

import gym

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from wrappers.utils import unwrap_wrapper_or_env


class CommandCenterRewardWrapper(gym.RewardWrapper):

    time_limit = 7 * 60     # seconds
    time_steps_per_second = CollectMineralAndGasEnv.max_game_step / time_limit

    def __init__(self, env: gym.Env, reward_diff: float = 10.0, time_margin: float = 100.0):
        super().__init__(env)
        self.source_env: Optional[CollectMineralAndGasEnv] = unwrap_wrapper_or_env(self.env, CollectMineralAndGasEnv)
        assert self.source_env, "CollectMineralAndGasEnv not found!"
        self.reward_diff = reward_diff
        self.time_margin = time_margin
        self.last_cc_built = False

    def reward(self, reward):
        if self.last_cc_built != self.source_env.cc_started:
            game_steps_left = CollectMineralAndGasEnv.max_game_step - self.source_env.get_game_step()
            seconds_left = game_steps_left / self.time_steps_per_second
            meaningful_seconds_left = seconds_left - self.time_margin
            reward += meaningful_seconds_left * self.reward_diff
        self.last_cc_built = self.source_env.cc_started
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_cc_built = self.source_env.cc_started
        return obs
