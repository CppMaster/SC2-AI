from typing import Optional

import gym

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from wrappers.utils import unwrap_wrapper_or_env


class SupplyTakenRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 100.0):
        super().__init__(env)
        self.source_env: Optional[CollectMineralAndGasEnv] = unwrap_wrapper_or_env(self.env, CollectMineralAndGasEnv)
        assert self.source_env, "CollectMineralAndGasEnv not found!"
        self.reward_diff = reward_diff
        self.last_supply = 0

    def reward(self, reward):
        supply = self.source_env.get_supply_taken()
        reward += (supply - self.last_supply) * self.reward_diff
        self.last_supply = supply
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_supply = self.source_env.get_supply_taken()
        return obs
