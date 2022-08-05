import logging
from typing import Optional

import gym

from minigames.simple_map.src.build_marines_env import BuildMarinesEnv
from wrappers.utils import unwrap_wrapper_or_env


class ScoreRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 0.01, kill_factor=2.0):
        super().__init__(env)
        self.source_env: Optional[BuildMarinesEnv] = unwrap_wrapper_or_env(self.env, BuildMarinesEnv)
        assert self.source_env, "BuildMarinesEnv not found!"
        self.logger = logging.getLogger("ScoreRewardWrapper")
        self.reward_diff = reward_diff
        self.kill_factor = kill_factor
        self.last_score = 0

    def reward(self, reward):
        score = self.get_summary_score()
        reward += (score - self.last_score) * self.reward_diff
        self.last_score = score
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_score = self.get_summary_score()
        return obs

    def get_summary_score(self) -> float:
        score_cumulative = self.source_env.get_score_cumulative()
        summary_score = score_cumulative.total_value_units
        summary_score += score_cumulative.total_value_structures
        summary_score += score_cumulative.killed_value_units * self.kill_factor
        summary_score += score_cumulative.killed_value_structures * self.kill_factor
        summary_score += score_cumulative.collected_minerals
        summary_score += score_cumulative.collected_vespene
        return summary_score
