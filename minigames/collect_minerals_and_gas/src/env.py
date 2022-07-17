import random
from enum import IntEnum
from typing import Optional, List, Set, Dict, Tuple, Any

import gym
from gym.spaces import Box, MultiDiscrete
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions, features
import numpy as np
import logging

from pysc2.lib.features import Player, UnitLayer, FeatureUnit
from pysc2.lib.units import Terran, Neutral


class ActionIndex(IntEnum):
    BUILD_SCV_1 = 0
    BUILD_SCV_2 = 1
    BUILD_CC = 2
    BUILD_SUPPLY = 3
    BUILD_REFINERY = 4


class ObservationIndex(IntEnum):
    MINERALS = 0  # scale 500
    SUPPLY_TAKEN = 1  # scale 50
    SUPPLY_ALL = 2  # scale 50
    SUPPLY_FREE = 3  # scale 16
    CC_BUILT = 4
    SCV_COUNT = 5  # scale 50
    REFINERY_COUNT = 6  # scale 4
    IS_REFINERY_BUILDING = 7
    TIME_LEFT = 8
    SUPPLY_DEPOT_COUNT = 9
    IS_SUPPLY_DEPOT_BUILDING = 10


class OrderId(IntEnum):
    HARVEST_MINERALS = 359
    HARVEST_RETURN = 362


class CollectMineralAndGasEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    supply_depot_locations = np.array([[29, 31], [29, 42], [31, 31], [31, 42], [33, 31], [33, 42], [35, 31], [35, 42]])
    cc_location = np.array([35, 36])
    max_game_step = 6720
    cc_optimal_workers = 16
    cc_max_workers = 24
    refinery_max_workers = 3

    def __init__(self, step_mul: int = 8, realtime: bool = False, random_order=False):
        self.settings = {
            'map_name': "CollectMineralsAndGas",
            'players': [sc2_env.Agent(sc2_env.Race.terran)],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                crop_to_playable_area=True
            ),
            'realtime': realtime,
            'step_mul': step_mul
        }
        self.action_space = MultiDiscrete([2] * len(ActionIndex))
        self.observation_space = Box(low=0.0, high=1.0, shape=(len(ObservationIndex),))
        self.env: Optional[SC2Env] = None
        self.logger = logging.getLogger("CollectMineralsAndGas")
        self.raw_obs = None
        self.random_order = random_order
        self.supply_depot_index = 0
        self.cc_started = False
        self.refinery_index = 0

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def step(self, action: Optional[np.ndarray] = None):
        self.raw_obs = self.env.step(self.get_actions(action))[0]
        derived_obs = self.get_derived_obs()
        return derived_obs, self.raw_obs.reward, self.raw_obs.last(), {}

    def get_actions(self, action: np.ndarray) -> List:
        mapped_actions = self.send_idle_workers_to_work()
        if action[ActionIndex.BUILD_SCV_1]:
            mapped_actions.append(self.build_scv(0))
        if action[ActionIndex.BUILD_SCV_2]:
            mapped_actions.append(self.build_scv(1))
        if action[ActionIndex.BUILD_SUPPLY]:
            mapped_actions.append(self.build_supply_depot())
        if action[ActionIndex.BUILD_CC]:
            mapped_actions.append(self.build_cc())
        if action[ActionIndex.BUILD_REFINERY]:
            mapped_actions.append(self.build_refinery())

        mapped_actions = list(filter(lambda x: x is not None, mapped_actions))
        if self.random_order:
            random.shuffle(mapped_actions)
        return mapped_actions

    def build_scv(self, ccidx: int):
        player = self.raw_obs.observation.player
        if player[Player.food_cap] - player[Player.food_used] < 1:
            return None
        if player[Player.minerals] < 50:
            return None

        ccs = self.get_units(Terran.CommandCenter)
        if ccidx >= len(ccs):
            return None
        ccs = sorted(ccs, key=lambda u: u["x"])
        cc = ccs[ccidx]
        if cc[FeatureUnit.order_length] == 0:
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now", ccs[ccidx].tag)
        return None

    def send_idle_workers_to_work(self) -> List:
        idle_scvs = list(filter(lambda u: u[FeatureUnit.order_length] == 0, self.get_units(Terran.SCV)))
        working_targets = self.get_working_targets(len(idle_scvs))
        orders = []
        for s_i, idle_scv in enumerate(idle_scvs):
            orders.append(actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", idle_scv[FeatureUnit.tag], working_targets[s_i][FeatureUnit.tag]
            ))
        return orders

    def get_working_targets(self, n_idle_workers: int) -> List:
        minerals = self.get_units(Neutral.MineralField)
        refineries = list(filter(lambda r: r[FeatureUnit.build_progress] == 100, self.get_units(Terran.Refinery)))
        left_minerals = list(filter(lambda m: m[FeatureUnit.x] <= 32, minerals))
        right_minerals = list(filter(lambda m: m[FeatureUnit.x] > 32, minerals))
        ccs = sorted(list(
            filter(lambda c: c[FeatureUnit.build_progress] == 100, self.get_units(Terran.CommandCenter))
        ), key=lambda c: c[FeatureUnit.x])
        cc_allocations = [c[FeatureUnit.assigned_harvesters] for c in ccs]
        refinery_allocations = [r[FeatureUnit.assigned_harvesters] for r in refineries]
        worker_targets: List = []
        for w_i in range(n_idle_workers):
            for c_i in range(len(ccs)):
                if cc_allocations[c_i] < self.cc_optimal_workers:
                    cc_allocations[c_i] += 1
                    worker_targets.append(random.choice(left_minerals if c_i == 0 else right_minerals))
                    break
            for r_i in range(len(refineries)):
                if refinery_allocations[r_i] < self.refinery_max_workers:
                    refinery_allocations[r_i] += 1
                    worker_targets.append(refineries[r_i])
                    break
            less_allocated_cc_index = np.argmin(cc_allocations)
            worker_targets.append(random.choice(left_minerals if less_allocated_cc_index == 0 else right_minerals))
        return worker_targets

    def build_supply_depot(self):
        if self.raw_obs.observation.player[Player.minerals] < 100:
            return None
        if self.supply_depot_index >= len(self.supply_depot_locations):
            return None
        location = self.supply_depot_locations[self.supply_depot_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None
        self.supply_depot_index += 1
        return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", worker[FeatureUnit.tag], location)

    def build_cc(self):
        if self.raw_obs.observation.player[Player.minerals] < 400:
            return None
        if self.cc_started:
            return None
        worker = self.get_nearest_worker(self.cc_location)
        self.cc_started = True
        return actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", worker[FeatureUnit.tag], self.cc_location)

    def build_refinery(self):
        if self.raw_obs.observation.player[Player.minerals] < 75:
            return None
        if self.refinery_index >= 4:
            return None
        geysers = sorted(self.get_units(Neutral.VespeneGeyser), key=lambda u: (u[FeatureUnit.x], u[FeatureUnit.y]))
        geyser = geysers[self.refinery_index]
        location = np.array([geyser[FeatureUnit.x], geyser[FeatureUnit.y]])
        worker = self.get_nearest_worker(location)
        self.refinery_index += 1
        return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", worker[FeatureUnit.tag], geyser[FeatureUnit.tag])

    def get_nearest_worker(self, location: np.ndarray):
        all_workers = self.get_units(Terran.SCV)
        idle_workers = list(filter(lambda u: u[FeatureUnit.order_length] == 0, all_workers))
        worker = self.get_nearest_worker_from_list(location, idle_workers)
        if worker is not None:
            return worker
        workers_mining = list(filter(lambda u: u[FeatureUnit.order_id_0] == OrderId.HARVEST_MINERALS, all_workers))
        worker = self.get_nearest_worker_from_list(location, workers_mining)
        if worker is not None:
            return worker
        return None

    @staticmethod
    def get_nearest_worker_from_list(location: np.ndarray, workers: List):
        if len(workers) == 0:
            return None
        worker_positions = np.array([[worker[FeatureUnit.x], worker[FeatureUnit.y]] for worker in workers])
        return workers[np.sum(np.power(worker_positions - location, 2), axis=1).argmin()]

    def reset(self):
        if self.env is None:
            self.init_env()

        self.supply_depot_index = 0
        self.cc_started = False
        self.refinery_index = 0
        self.raw_obs = self.env.reset()[0]
        return self.get_derived_obs()

    def get_derived_obs(self) -> np.ndarray:
        obs = np.zeros(shape=self.observation_space.shape)
        player = self.raw_obs.observation.player
        obs[ObservationIndex.MINERALS] = player[Player.minerals] / 500
        obs[ObservationIndex.SUPPLY_TAKEN] = player[Player.food_used] / 50
        obs[ObservationIndex.SUPPLY_ALL] = player[Player.food_cap] / 50
        obs[ObservationIndex.SUPPLY_FREE] = (player[Player.food_cap] - player[Player.food_used]) / 16
        obs[ObservationIndex.CC_BUILT] = float(self.cc_started)
        obs[ObservationIndex.SCV_COUNT] = len(self.get_units(Terran.SCV)) / 50
        obs[ObservationIndex.REFINERY_COUNT] = self.refinery_index / 4
        obs[ObservationIndex.IS_REFINERY_BUILDING] = float(sum(
            [refinery[FeatureUnit.build_progress] < 100 for refinery in self.get_units(Terran.Refinery)]
        ))
        obs[ObservationIndex.TIME_LEFT] = (self.max_game_step - self.raw_obs.observation.game_loop) / self.max_game_step
        obs[ObservationIndex.SUPPLY_DEPOT_COUNT] = self.supply_depot_index / 8
        obs[ObservationIndex.IS_SUPPLY_DEPOT_BUILDING] = self.get_supply_depots_in_progress()
        return obs

    def get_units(self, unit_type: int):
        return [unit for unit in self.raw_obs.observation.raw_units if unit.unit_type == unit_type]

    def render(self, mode="human"):
        pass

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def get_invalid_actions(self) -> Set[int]:
        return {index for index, valid in self.get_action_to_valid().items() if not valid}

    def is_2nd_cc_built(self) -> bool:
        ccs = sorted(self.get_units(Terran.CommandCenter), key=lambda c: c[FeatureUnit.x])
        return len(ccs) == 2 and ccs[1][FeatureUnit.build_progress] == 100

    def get_action_to_valid(self) -> Dict[int, bool]:
        player = self.raw_obs.observation.player
        food_free = player[Player.food_cap] - player[Player.food_used]
        second_cc_build = self.is_2nd_cc_built()
        place_to_supply_depot = self.supply_depot_index < len(self.supply_depot_locations)
        return {
            ActionIndex.BUILD_SCV_1: player.minerals >= 50 and food_free >= 1,
            ActionIndex.BUILD_SCV_2: player.minerals >= 50 and food_free >= 1 and second_cc_build,
            ActionIndex.BUILD_CC: player.minerals >= 400 and not self.cc_started,
            ActionIndex.BUILD_SUPPLY: player.minerals >= 100 and place_to_supply_depot,
            ActionIndex.BUILD_REFINERY: player.minerals >= 75 and self.refinery_index < 4,
            len(ActionIndex): True
        }

    def get_mineral_workers(self) -> int:
        return sum([
            min(cc[FeatureUnit.assigned_harvesters], self.cc_optimal_workers)
            for cc in self.get_units(Terran.CommandCenter)
        ])

    def get_lesser_mineral_workers(self) -> int:
        return sum([
            max(min(cc[FeatureUnit.assigned_harvesters], self.cc_max_workers) - self.cc_optimal_workers, 0)
            for cc in self.get_units(Terran.CommandCenter)
        ])

    def get_gas_workers(self) -> int:
        return sum([
            min(r[FeatureUnit.assigned_harvesters], self.refinery_max_workers)
            for r in self.get_units(Terran.Refinery)
        ])

    def get_supply_taken(self) -> int:
        return self.raw_obs.observation.player[Player.food_used]

    def get_supply_cap(self) -> int:
        return self.raw_obs.observation.player[Player.food_cap]

    def get_supply_depots_in_progress(self) -> int:
        return sum(
            [supply_depot[FeatureUnit.build_progress] < 100 for supply_depot in self.get_units(Terran.SupplyDepot)]
        )

    def get_game_step(self) -> int:
        return self.raw_obs.observation.game_loop
