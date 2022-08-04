import logging
from typing import Optional

import gym
from pysc2.lib.units import Terran

from minigames.simple_map.src.build_marines_env import BuildMarinesEnv
from wrappers.utils import unwrap_wrapper_or_env


class ScoreRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 0.01):
        super().__init__(env)
        self.source_env: Optional[BuildMarinesEnv] = unwrap_wrapper_or_env(self.env, BuildMarinesEnv)
        assert self.source_env, "BuildMarinesEnv not found!"
        self.logger = logging.getLogger("ScoreRewardWrapper")
        self.reward_diff = reward_diff
        self.last_score = 0

    def reward(self, reward):
        score = self.source_env.get_score()
        reward += (score - self.last_score) * self.reward_diff
        self.last_score = score
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_score = self.source_env.get_score()
        return obs
