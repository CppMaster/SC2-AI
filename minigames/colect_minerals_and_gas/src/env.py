import random
from enum import IntEnum
from typing import Optional, List

import gym
from gym.spaces import Box, MultiDiscrete
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions, features
import numpy as np
import logging

from pysc2.lib.features import Dimensions, Player, UnitLayer, FeatureUnit
from pysc2.lib.units import Terran, Neutral


class ActionIndex(IntEnum):
    BUILD_SCV_1 = 0
    BUILD_SCV_2 = 1
    BUILD_CC = 2
    BUILD_SUPPLY = 3
    BUILD_REFINERY = 4


class ObservationIndex(IntEnum):
    MINERALS = 0    # scale 500
    SUPPLY_TAKEN = 1    # scale 50
    SUPPLY_ALL = 2      # scale 50
    SUPPLY_FREE = 3     # scale 16
    CC_BUILT = 4
    SCV_COUNT = 5   # scale 50
    REFINERY_COUNT = 6  # scale 4
    IS_REFINERY_BUILDING = 7


class CollectMineralAndGasEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, step_mul: int = 8, realtime: bool = False, resolution: int = 32, random_order=False):
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
        self.action_space = MultiDiscrete([2] * len(ActionIndex))
        self.observation_space = Box(low=0.0, high=1.0, shape=(len(ObservationIndex), ))
        self.env: Optional[SC2Env] = None
        self.logger = logging.getLogger("CollectMineralsAndGas")
        self.raw_obs = None
        self.random_order = random_order

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def step(self, action: Optional[np.ndarray] = None):
        self.raw_obs = self.env.step(self.get_actions(action))[0]
        derived_obs = self.get_derived_obs()
        return derived_obs, self.raw_obs.reward, self.raw_obs.last(), {}

    def get_actions(self, action: np.ndarray) -> List:
        mapped_actions = self.send_idle_workers_to_minerals()
        if action[ActionIndex.BUILD_SCV_1]:
            mapped_actions.append(self.build_scv(0))
        if action[ActionIndex.BUILD_SCV_2]:
            mapped_actions.append(self.build_scv(1))
        '''
        TODO fill mapped_actions
        build cc
        build refinery
        build supply depot
        '''
        mapped_actions = list(filter(lambda x: x is not None, mapped_actions))
        if self.random_order:
            random.shuffle(mapped_actions)
        return mapped_actions

    def build_scv(self, ccidx: int):
        player = self.raw_obs.observation.player
        if player[Player.food_cap] - player[Player.food_used] < 1:
            return None

        ccs = self.get_units(Terran.CommandCenter)
        if ccidx >= len(ccs):
            return None
        ccs = sorted(ccs, key=lambda u: u["x"])
        cc = ccs[ccidx]
        if cc[FeatureUnit.order_length] == 0:
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now", ccs[ccidx].tag)
        return None

    def send_idle_workers_to_minerals(self) -> List:
        idle_scvs = list(filter(lambda u: u[FeatureUnit.order_length] == 0, self.get_units(Terran.SCV)))
        minerals = self.get_units(Neutral.MineralField)
        mineral_positions = np.array([[mineral[FeatureUnit.x], mineral[FeatureUnit.y]] for mineral in minerals])
        orders = []
        for idle_scv in idle_scvs:
            scv_position = np.array([idle_scv[FeatureUnit.x], idle_scv[FeatureUnit.y]])
            mineral_idx = np.sum(np.power(mineral_positions - scv_position, 2), axis=1).argmin()
            orders.append(actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", idle_scv[FeatureUnit.tag], minerals[mineral_idx][FeatureUnit.tag]
            ))
        return orders

    def reset(self):
        if self.env is None:
            self.init_env()

        self.raw_obs = self.env.reset()[0]
        return self.get_derived_obs()

    def get_derived_obs(self) -> np.ndarray:
        obs = np.zeros(shape=self.observation_space.shape)
        player = self.raw_obs.observation.player
        obs[ObservationIndex.MINERALS] = player[Player.minerals] / 500
        obs[ObservationIndex.SUPPLY_TAKEN] = player[Player.food_used] / 50
        obs[ObservationIndex.SUPPLY_ALL] = player[Player.food_cap] / 50
        obs[ObservationIndex.SUPPLY_FREE] = (player[Player.food_cap] - player[Player.food_used]) / 16
        obs[ObservationIndex.CC_BUILT] = float(len(self.get_units(Terran.CommandCenter)) > 1)
        obs[ObservationIndex.SCV_COUNT] = len(self.get_units(Terran.SCV)) / 50
        obs[ObservationIndex.REFINERY_COUNT] = len(self.get_units(Terran.Refinery)) / 4
        obs[ObservationIndex.IS_REFINERY_BUILDING] = float(any(
            [refinery[UnitLayer.build_progress < 100] for refinery in self.get_units(Terran.Refinery)]
        ))
        return obs

    def get_units(self, unit_type: int):
        return [unit for unit in self.raw_obs.observation.raw_units if unit.unit_type == unit_type]

    def render(self, mode="human"):
        pass

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()
