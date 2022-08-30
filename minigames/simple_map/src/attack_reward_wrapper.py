import logging
from typing import Optional, Dict

import gym
import numpy as np
from gym.spaces import Discrete

from minigames.simple_map.src.build_marines_env import BuildMarinesEnv, ActionIndex
from wrappers.utils import unwrap_wrapper_or_env


class AttackRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, reward_diff: float = 1.0, time_offset=0.2,
                 custom_multipliers: Optional[Dict[int, float]] = None, action_penalty: float = 0.2):
        super().__init__(env)
        self.source_env: Optional[BuildMarinesEnv] = unwrap_wrapper_or_env(self.env, BuildMarinesEnv)
        assert self.source_env, "BuildMarinesEnv not found!"
        assert isinstance(env.action_space, Discrete), "Only Discrete action space handled"
        self.logger = logging.getLogger("AttackRewardWrapper")
        self.reward_diff = reward_diff
        self.time_offset = time_offset
        self.last_command = len(ActionIndex)
        self.current_command = len(ActionIndex)
        custom_multipliers = custom_multipliers or {}
        self.multipliers = {
            ActionIndex.ATTACK: 1.0,
            ActionIndex.RETREAT: -1.0,
            ActionIndex.STOP_ARMY: -0.2,
            ActionIndex.GATHER_ARMY: 0.5,
            len(ActionIndex): 0.0
        } | custom_multipliers
        self.action_penalty = action_penalty

    def reward(self, reward):
        time = self.source_env.get_normalized_time() - self.time_offset
        multiplier = self.multipliers[self.current_command] - self.multipliers[self.last_command]
        multiplier -= self.action_penalty
        shaped_reward = time * multiplier * self.reward_diff
        if shaped_reward:
            self.logger.debug(f"Shaped reward: {shaped_reward}n\ttime: {time},\tmultiplier: {multiplier},\t"
                              f"last command: {ActionIndex.int_to_name(self.last_command)},\t"
                              f"current command: {ActionIndex.int_to_name(self.current_command)}")
        return reward + shaped_reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_command = len(ActionIndex)
        self.current_command = len(ActionIndex)
        return obs

    def step(self, action: int):
        observation, reward, done, info = self.env.step(action)
        if self.is_army_action(action):
            self.current_command = action
        reward = self.reward(reward)
        if self.is_army_action(action):
            self.last_command = action
        return observation, reward, done, info

    def is_army_action(self, action: int) -> bool:
        return action in self.source_env.army_actions and self.source_env.has_any_military_units()

