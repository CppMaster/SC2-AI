from typing import Optional

import gym

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from wrappers.utils import unwrap_wrapper_or_env


class SupplyDepotRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 100.0, free_supply_margin: int = 6):
        super().__init__(env)
        self.source_env: Optional[CollectMineralAndGasEnv] = unwrap_wrapper_or_env(self.env, CollectMineralAndGasEnv)
        assert self.source_env, "CollectMineralAndGasEnv not found!"
        self.reward_diff = reward_diff
        self.free_supply_margin = free_supply_margin
        self.last_supply_depot_index = 0

    def reward(self, reward):
        supply_depot_index = self.source_env.supply_depot_index
        if supply_depot_index > self.last_supply_depot_index:
            expected_supply_cap = self.source_env.get_supply_cap() + self.source_env.get_supply_depots_in_progress() * 8
            supply_taken = self.source_env.get_supply_taken() + self.free_supply_margin
            reward += (supply_taken - expected_supply_cap) * self.reward_diff
        self.last_supply_depot_index = supply_depot_index
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_supply_depot_index = self.source_env.supply_depot_index
        return obs
