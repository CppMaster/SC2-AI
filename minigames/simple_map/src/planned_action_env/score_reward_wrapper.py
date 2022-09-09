import logging
import numpy as np

from minigames.simple_map.src.planned_action_env.reward_shaper import RewardShaper


class ScoreRewardShaper(RewardShaper):

    def __init__(self, reward_diff: float = 0.001, kill_factor: float = 1.0,
                 army_factor: float = 1.0, mined_factor: float = 0.1, economy_factor: float = 0.1):
        super().__init__()
        self.logger = logging.getLogger("ScoreRewardShaper")
        self.last_score = 0
        self.reward_diff = reward_diff
        self.kill_factor = kill_factor
        self.army_factor = army_factor
        self.mined_factor = mined_factor
        self.economy_factor = economy_factor

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

        collected_resources = \
            (score_cumulative.collected_minerals + score_cumulative.collected_vespene) * self.mined_factor
        used_army = \
            (score_by_category.used_minerals.army + score_by_category.used_vespene.army) * self.army_factor
        used_economy = \
            (score_by_category.used_minerals.economy + score_by_category.used_vespene.economy) * self.economy_factor
        killed = \
            np.sum(np.array(score_by_category.killed_minerals)) + np.sum(np.array(score_by_category.killed_vespene)) \
            * self.kill_factor

        summary_score = 0.0
        summary_score += collected_resources
        summary_score += used_army
        summary_score += used_economy
        summary_score += killed

        self.logger.debug(
            f"Score cumulative:\t"
            f"collected_resources: {collected_resources}\t"
            f"used_army: {used_army}\t"
            f"used_economy: {used_economy}\t"
            f"killed: {killed}\t"
        )
        return summary_score
