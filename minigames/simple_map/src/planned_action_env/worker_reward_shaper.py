import logging

from pysc2.lib.units import Terran

from minigames.simple_map.src.planned_action_env.env import ActionIndex
from minigames.simple_map.src.planned_action_env.reward_shaper import RewardShaper


class WorkerRewardShaper(RewardShaper):

    def __init__(self, reward_diff: float = 0.01, optimal_reward: float = 1.0, suboptimal_reward: float = 0.0,
                 over_max_reward: float = -1.0):
        super().__init__()
        self.logger = logging.getLogger("WorkerRewardShaper")
        self.reward_diff = reward_diff
        self.optimal_reward = optimal_reward
        self.suboptimal_reward = suboptimal_reward
        self.over_max_reward = over_max_reward
        self.last_rl_step = 0

    def reset(self):
        self.last_rl_step = 0

    def get_shaped_reward(self) -> float:
        reward = 0.0
        if self.env.last_action_index == ActionIndex.BUILD_SCV and self.last_rl_step != self.env.rl_step:
            n_cc = len(self.env.get_units(Terran.CommandCenter, alliance=1))
            optimal_scv_count = self.env.cc_optimal_workers * n_cc
            max_scv_count = self.env.cc_max_workers * n_cc
            current_scv_count = len(self.env.get_units(Terran.SCV, alliance=1))
            if current_scv_count < optimal_scv_count:
                reward = self.optimal_reward
            elif current_scv_count < max_scv_count:
                reward = self.suboptimal_reward
            else:
                reward = self.over_max_reward
            self.logger.debug(f"Current SCV count: {current_scv_count},\treward: {reward}")

        self.last_rl_step = self.env.rl_step
        return reward
