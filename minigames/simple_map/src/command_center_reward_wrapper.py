from typing import Optional

import gym

from minigames.simple_map.src.build_marines_env import BuildMarinesEnv
from wrappers.utils import unwrap_wrapper_or_env


class CommandCenterRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 10.0):
        super().__init__(env)
        self.source_env: Optional[BuildMarinesEnv] = unwrap_wrapper_or_env(self.env, BuildMarinesEnv)
        assert self.source_env, "BuildMarinesEnv not found!"
        self.reward_diff = reward_diff
        self.last_cc_built = False

    def reward(self, reward):
        if self.last_cc_built != self.source_env.cc_started:
            reward += self.reward_diff
        self.last_cc_built = self.source_env.cc_started
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_cc_built = self.source_env.cc_started
        return obs
