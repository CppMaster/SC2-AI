from typing import Optional

import gym
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions, features, units
from gym import spaces
import numpy as np
import logging


class MoveToBeaconEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, step_mul: int = 8, raw_resolution: int = 64, realtime: bool = False):
        self.settings = {
            'map_name': "MoveToBeacon",
            'players': [sc2_env.Agent(sc2_env.Race.terran)],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=raw_resolution),
            'realtime': realtime,
            'step_mul': step_mul
        }
        self.raw_resolution = raw_resolution
        self.action_space = spaces.Box(low=0, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,))
        self.env: Optional[SC2Env] = None
        self.unit_tag = None
        self.logger = logging.getLogger("MoveToBeaconEnv")

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def step(self, action: np.ndarray):
        raw_obs = self.env.step([actions.RAW_FUNCTIONS.Move_pt("now", self.unit_tag, action * self.raw_resolution)])[0]
        self.update_unit_tag(raw_obs)
        derived_obs = self.get_derived_obs(raw_obs)
        self.logger.debug(f"Action: {action}, Reward: {raw_obs.reward}, Obs: {derived_obs}")
        return derived_obs, raw_obs.reward, raw_obs.last(), {}

    def reset(self):
        if self.env is None:
            self.init_env()

        raw_obs = self.env.reset()[0]
        self.update_unit_tag(raw_obs)
        return self.get_derived_obs(raw_obs)

    def get_derived_obs(self, raw_obs):
        unit = self.get_unit(raw_obs)
        target = self.get_target(raw_obs)
        return np.array([unit.x, unit.y, target.x, target.y]) / self.raw_resolution

    def render(self, mode="human"):
        pass

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    @staticmethod
    def get_unit(raw_obs):
        return [unit for unit in raw_obs.observation.raw_units if unit.unit_type == units.Terran.Marine][0]

    @staticmethod
    def get_target(raw_obs):
        return [unit for unit in raw_obs.observation.raw_units if unit.unit_type == 317][0]

    def update_unit_tag(self, raw_obs):
        self.unit_tag = self.get_unit(raw_obs).tag
