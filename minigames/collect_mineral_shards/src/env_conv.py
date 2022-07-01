from typing import Optional, List

import gym
from gym.spaces import Box
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions, features, units
import numpy as np
import logging

from pysc2.lib.features import Dimensions


class CollectMineralShardsConvEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, step_mul: int = 8, realtime: bool = False, resolution: int = 32):
        self.settings = {
            'map_name': "CollectMineralShards",
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
        self.action_space = Box(low=0.0, high=1.0, shape=(resolution, resolution, 2))
        self.observation_space = Box(low=0.0, high=1.0, shape=(resolution, resolution, 3))
        self.env: Optional[SC2Env] = None
        self.logger = logging.getLogger("CollectMineralShardsEnv")
        self.unit_tags: List[int] = []

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def step(self, action: np.ndarray):
        mapped_actions = []
        for idx, tag in enumerate(self.unit_tags):
            pos = np.array(np.unravel_index(np.argmax(action[:, :, idx], axis=None), action[:, :, idx].shape))
            mapped_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", tag, pos))
        raw_obs = self.env.step(mapped_actions)[0]
        self.update_unit_tag(raw_obs)
        derived_obs = self.get_derived_obs(raw_obs)
        return derived_obs, raw_obs.reward, raw_obs.last(), {}

    def reset(self):
        if self.env is None:
            self.init_env()

        raw_obs = self.env.reset()[0]
        self.update_unit_tag(raw_obs)
        return self.get_derived_obs(raw_obs)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros(shape=self.observation_space.shape)
        obs[:, :, 0] = raw_obs.observation.feature_screen["unit_type"] == 1680
        player_units = self.get_units(raw_obs)
        marine_positions = np.array([
            [player_units[0]["x"], player_units[0]["y"]],
            [player_units[1]["x"], player_units[1]["y"]]
        ])
        for idx, pos in enumerate(marine_positions):
            obs[pos[0], pos[1], 1 + idx] = 1.0
        return obs

    @staticmethod
    def get_units(raw_obs):
        return [unit for unit in raw_obs.observation.raw_units if unit.unit_type == units.Terran.Marine]

    def update_unit_tag(self, raw_obs):
        self.unit_tags = [unit.tag for unit in self.get_units(raw_obs)]

    def render(self, mode="human"):
        pass

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()
