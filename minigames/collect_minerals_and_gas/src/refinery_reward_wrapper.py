from typing import Optional

import gym
from pysc2.lib.units import Terran

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from wrappers.utils import unwrap_wrapper_or_env


class RefineryRewardWrapper(gym.RewardWrapper):

    time_limit = 7 * 60     # seconds
    time_steps_per_second = CollectMineralAndGasEnv.max_game_step / time_limit

    def __init__(self, env: gym.Env, reward_diff: float = 100.0, workers_slots_margin: float = 4.,
                 suboptimal_worker_slot_weight: float = 0.5):
        super().__init__(env)
        self.source_env: Optional[CollectMineralAndGasEnv] = unwrap_wrapper_or_env(self.env, CollectMineralAndGasEnv)
        assert self.source_env, "CollectMineralAndGasEnv not found!"
        self.reward_diff = reward_diff
        self.workers_slots_margin = workers_slots_margin
        self.last_refinery_index = 0
        self.suboptimal_worker_slot_weight = suboptimal_worker_slot_weight

    def reward(self, reward):
        if self.last_refinery_index != self.source_env.refinery_index:
            worker_slots, suboptimal_worker_slots = self.source_env.get_worker_slots()
            n_future_refineries = self.source_env.refinery_index - len(self.source_env.get_units(Terran.Refinery))
            worker_slots_score = worker_slots
            worker_slots_score += suboptimal_worker_slots * self.suboptimal_worker_slot_weight
            worker_slots_score += n_future_refineries * self.source_env.refinery_max_workers
            worker_slots_score -= self.workers_slots_margin
            reward -= worker_slots_score * self.reward_diff
        self.last_refinery_index = self.source_env.refinery_index
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_refinery_index = self.last_refinery_index
        return obs
