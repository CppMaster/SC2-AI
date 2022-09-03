

import logging
from typing import Optional

import gym
import numpy as np

from minigames.simple_map.src.planned_action_env.reward_shaper import RewardShaper


class ScoreRewardShaper(RewardShaper):

    def __init__(self, reward_diff: float = 0.001, kill_factor: float = 1.0,
                 army_bonus_factor: float = 1.0):
        super().__init__()
        self.logger = logging.getLogger("ScoreRewardShaper")
        self.last_score = 0
        self.reward_diff = reward_diff
        self.kill_factor = kill_factor
        self.army_bonus_factor = army_bonus_factor

    def get_shaped_reward(self) -> float:
        score = self.get_summary_score()
        reward_delta = (score - self.last_score) * self.reward_diff
        self.logger.debug(f"Score: {score},\tReward delta: {reward_delta}")
        self.last_score = score
        return reward_delta

    def reward(self, reward):
        score = self.get_summary_score()
        reward_delta = (score - self.last_score) * self.reward_diff
        self.logger.debug(f"Score: {score},\tReward delta: {reward_delta}")
        reward += reward_delta
        self.last_score = score
        return reward

    def reset(self):
        self.last_score = self.get_summary_score()

    def get_summary_score(self) -> float:
        score_cumulative = self.env.get_score_cumulative()
        score_by_category = self.env.get_score_by_category()

        collected_minerals = score_cumulative.collected_minerals
        collected_vespene = score_cumulative.collected_vespene
        used_minerals = np.sum(np.array(score_by_category.used_minerals))
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
        army_bonus = (score_by_category.used_minerals.army + score_by_category.used_vespene.army) \
                     * self.army_bonus_factor
        summary_score += army_bonus

        self.logger.debug(
            f"Score cumulative:\t"
            f"collected_minerals: {collected_minerals}\t"
            f"collected_vespene: {collected_vespene}\t"
            f"used_minerals: {used_minerals}\t"
            f"used_vespene: {used_vespene}\t"
            f"killed_vespene: {killed_minerals * self.kill_factor}\t"
            f"killed_value_structures: {killed_vespene * self.kill_factor}\t"
            f"army_bonus: {army_bonus}"
        )
        return summary_score
