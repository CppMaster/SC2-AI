import logging
from typing import Optional

import gym
import numpy as np

from minigames.simple_map.src.instant_action_env.build_marines_env import BuildMarinesEnv
from utils.plot.real_time_plot import RealTimePlot
from wrappers.utils import unwrap_wrapper_or_env


class ScoreRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 0.01, kill_factor=1.0, draw_plot=False, mined_factor=1.0):
        super().__init__(env)
        self.source_env: Optional[BuildMarinesEnv] = unwrap_wrapper_or_env(self.env, BuildMarinesEnv)
        assert self.source_env, "BuildMarinesEnv not found!"
        self.logger = logging.getLogger("ScoreRewardWrapper")
        self.reward_diff = reward_diff
        self.kill_factor = kill_factor
        self.mined_factor = mined_factor
        self.last_score = 0
        self.logger = logging.getLogger(__name__)
        self.reward_delta_plot: Optional[RealTimePlot]
        if draw_plot:
            self.reward_delta_plot = RealTimePlot()
        else:
            self.reward_delta_plot = None

    def reward(self, reward):
        score = self.get_summary_score()
        reward_delta = (score - self.last_score) * self.reward_diff
        self.logger.debug(f"Score: {score},\tReward delta: {reward_delta}")
        if self.reward_delta_plot:
            self.reward_delta_plot.update(reward_delta)
        reward += reward_delta
        self.last_score = score
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_score = self.get_summary_score()
        if self.reward_delta_plot:
            self.reward_delta_plot.reset()
        return obs

    def get_summary_score(self) -> float:
        score_cumulative = self.source_env.get_score_cumulative()
        score_by_category = self.source_env.get_score_by_category()

        collected_minerals = score_cumulative.collected_minerals * self.mined_factor
        collected_vespene = score_cumulative.collected_vespene * self.mined_factor
        used_minerals = score_by_category.used_minerals.army
        used_vespene = np.sum(np.array(score_by_category.used_vespene))
        killed_minerals = np.sum(np.array(score_by_category.killed_minerals))
        killed_vespene = np.sum(np.array(score_by_category.killed_vespene))

        summary_score = 0.0
        summary_score += collected_minerals
        summary_score += collected_vespene
        summary_score += used_minerals
        summary_score += used_vespene
        summary_score += killed_minerals * self.kill_factor
        summary_score += killed_vespene * self.kill_factor

        self.logger.debug(
            f"Score cumulative:\t"
            f"collected_minerals: {collected_minerals}\t"
            f"collected_vespene: {collected_vespene}\t"
            f"used_minerals: {used_minerals}\t"
            f"used_vespene: {used_vespene}\t"
            f"killed_vespene: {killed_minerals * self.kill_factor}\t"
            f"killed_value_structures: {killed_vespene * self.kill_factor}\t"
        )
        return summary_score
