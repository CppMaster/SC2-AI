from typing import Optional, List

import gym
from gym.spaces import Dict, Box
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions, features, units
import numpy as np
import logging

from pysc2.lib.features import Dimensions


class CollectMineralAndGasEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, step_mul: int = 8, realtime: bool = False, resolution: int = 32):
        self.settings = {
            'map_name': "CollectMineralsAndGas",
            'players': [sc2_env.Agent(sc2_env.Race.terran)],
            'agent_interface_format': features.AgentInterfaceFormat(
                feature_dimensions=Dimensions(screen=(resolution, resolution), minimap=(resolution, resolution)),
                raw_resolution=resolution,
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                crop_to_playable_area=True
            ),
            'realtime': realtime,
            'step_mul': step_mul
        }
        self.resolution = resolution
        # TODO action_space and observation_space
        self.action_space = Box(low=-0.5, high=0.5, shape=(4,))
        self.observation_space = Dict({
            "minerals": Box(low=-0.5, high=0.5, shape=(resolution, resolution)),
            "marines": Box(low=-0.5, high=0.5, shape=(2, 2))
        })
        self.env: Optional[SC2Env] = None
        self.logger = logging.getLogger("CollectMineralsAndGas")
        self.last_raw_obs = None

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def step(self, action: Optional[np.ndarray] = None):
        self.last_raw_obs = self.env.step(self.get_actions(action))[0]
        derived_obs = self.get_derived_obs(self.last_raw_obs)
        return derived_obs, self.last_raw_obs.reward, self.last_raw_obs.last(), {}

    def get_actions(self, action: np.ndarray) -> List:
        mapped_actions = []
        # TODO fill mapped_actions
        return mapped_actions

    def reset(self):
        if self.env is None:
            self.init_env()

        self.last_raw_obs = self.env.reset()[0]
        return self.get_derived_obs(self.last_raw_obs)

    def get_derived_obs(self, raw_obs):
        # TODO return derived obs
        pass

    @staticmethod
    def get_units(raw_obs, unit_type: int):
        return [unit for unit in raw_obs.observation.raw_units if unit.unit_type == unit_type]

    def render(self, mode="human"):
        pass

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()
