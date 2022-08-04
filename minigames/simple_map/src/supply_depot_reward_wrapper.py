import logging
from typing import Optional

import gym
from pysc2.lib.units import Terran

from minigames.simple_map.src.build_marines_env import BuildMarinesEnv
from wrappers.utils import unwrap_wrapper_or_env


class SupplyDepotRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 10.0, free_supply_margin_factor: float = 1.0):
        super().__init__(env)
        self.source_env: Optional[BuildMarinesEnv] = unwrap_wrapper_or_env(self.env, BuildMarinesEnv)
        assert self.source_env, "BuildMarinesEnv not found!"
        self.logger = logging.getLogger("SupplyDepotRewardWrapper")
        self.reward_diff = reward_diff
        self.free_supply_margin_factor = free_supply_margin_factor
        self.last_supply_depot_index = 0

    def reward(self, reward):
        supply_depot_index = self.source_env.supply_depot_index
        if supply_depot_index > self.last_supply_depot_index:
            expected_supply_cap = self.source_env.get_expected_supply_cap() - 8
            supply_taken = self.source_env.get_supply_taken() + self.free_supply_margin_factor * (
                    1 + len(self.source_env.get_units(Terran.Barracks))
            )
            shaped_reward = (supply_taken - expected_supply_cap) * self.reward_diff
            reward += shaped_reward
            self.logger.debug(f"Supply taken: {supply_taken}, Expected supply cap: {expected_supply_cap}, "
                              f"Shaped reward: {shaped_reward}")
        self.last_supply_depot_index = supply_depot_index
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_supply_depot_index = self.source_env.supply_depot_index
        return obs
