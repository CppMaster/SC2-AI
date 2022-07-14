from typing import Optional

import gym

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from wrappers.utils import unwrap_wrapper_or_env


class WorkersActiveRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, warmup_steps: int = 12, mineral_reward: float = 100.0,
                 lesser_mineral_reward: float = 50.0, gas_reward: float = 75.0):
        super().__init__(env)
        self.source_env: Optional[CollectMineralAndGasEnv] = unwrap_wrapper_or_env(self.env, CollectMineralAndGasEnv)
        assert self.source_env, "CollectMineralAndGasEnv not found!"
        self.warmup_steps = warmup_steps
        self.mineral_reward = mineral_reward
        self.lesser_mineral_reward = lesser_mineral_reward
        self.gas_reward = gas_reward

        self.current_step = 0
        self.last_mineral_workers = 0
        self.last_lesser_mineral_workers = 0
        self.last_gas_workers = 0

    def reward(self, reward):
        mineral_workers = self.source_env.get_mineral_workers()
        lesser_mineral_workers = self.source_env.get_lesser_mineral_workers()
        gas_workers = self.source_env.get_gas_workers()

        if self.current_step >= self.warmup_steps:
            reward += (mineral_workers - self.last_mineral_workers) * self.mineral_reward
            reward += (lesser_mineral_workers - self.last_lesser_mineral_workers) * self.lesser_mineral_reward
            reward += (gas_workers - self.last_gas_workers) * self.gas_reward

        self.last_mineral_workers = mineral_workers
        self.last_lesser_mineral_workers = lesser_mineral_workers
        self.last_gas_workers = gas_workers

        self.current_step += 1

        return reward

    def reset(self, **kwargs):
        self.current_step = 0
        self.last_mineral_workers = 0
        self.last_lesser_mineral_workers = 0
        self.last_gas_workers = 0
        return self.env.reset(**kwargs)
