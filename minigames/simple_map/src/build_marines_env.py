import logging
from enum import IntEnum
import random
from typing import Optional, List, Set, Union

import gym
import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env, Dimensions
from pysc2.lib import features, actions
from pysc2.lib.features import FeatureUnit, Player
from pysc2.lib.units import Terran, Neutral

from minigames.collect_minerals_and_gas.src.env import OrderId


class ActionIndex(IntEnum):
    BUILD_MARINE = 0
    BUILD_SCV = 1
    BUILD_SUPPLY = 2
    BUILD_BARRACKS = 3
    ATTACK = 4


class ObservationIndex(IntEnum):
    MINERALS = 0        # scale 500
    SUPPLY_TAKEN = 1    # scale 150
    SUPPLY_ALL = 2      # scale 150
    SUPPLY_FREE = 3     # scale 16
    SCV_COUNT = 4       # scale 50
    TIME_STEP = 5       # scale 20000
    SUPPLY_DEPOT_COUNT = 6
    IS_SUPPLY_DEPOT_BUILDING = 7
    BARRACKS_COUNT = 8
    IS_BARRACKS_BUILDING = 9
    CAN_BUILD_SCV = 10
    CAN_BUILD_BARRACKS = 11
    CAN_BUILD_SUPPLY_DEPOT = 12
    CAN_BUILD_MARINE = 13
    SCV_IN_PROGRESS = 14
    MARINES_IN_PROGRESS = 15


class BuildMarinesEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    scv_limit = 50
    rally_position = np.array([20, 36])
    map_dimensions = (88, 96)
    base_locations = [(26, 25), None, None, (54, 68)]

    def __init__(self, step_mul: int = 8, realtime: bool = False, is_discrete: bool = True):
        self.settings = {
            'map_name': "Simple64",
            'players': [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                feature_dimensions=Dimensions(screen=self.map_dimensions, minimap=self.map_dimensions),
                use_feature_units=True,
                crop_to_playable_area=False
            ),
            'realtime': realtime,
            'step_mul': step_mul
        }

        self.is_discrete = is_discrete
        self.action_space: Union[Discrete, MultiDiscrete]
        if self.is_discrete:
            self.action_space = Discrete(len(ActionIndex) + 1)
        else:
            self.action_space = MultiDiscrete([2] * len(ActionIndex))

        self.observation_space = Box(low=0.0, high=1.0, shape=(len(ObservationIndex),))
        self.env: Optional[SC2Env] = None
        self.logger = logging.getLogger("BuildMarinesEnv")
        self.raw_obs = None

        self.supply_depot_index = 0
        self.barracks_index = 0
        self.rallies_set: Set[int] = set()
        self.player_on_left = False
        self.supply_depot_locations = np.zeros(shape=(0, 2))
        self.barracks_locations = np.zeros(shape=(0, 2))
        self.enemy_base_location = np.zeros(shape=(2, ))

        self.action_mapping = {
            ActionIndex.BUILD_MARINE: self.build_marine,
            ActionIndex.BUILD_SCV: self.build_scv,
            ActionIndex.BUILD_SUPPLY: self.build_supply_depot,
            ActionIndex.BUILD_BARRACKS: self.build_barracks,
            ActionIndex.ATTACK: self.attack
        }
        self.valid_action_mapping = {
            ActionIndex.BUILD_MARINE: self.can_build_marine,
            ActionIndex.BUILD_SCV: self.can_build_scv,
            ActionIndex.BUILD_SUPPLY: self.can_build_supply_depot,
            ActionIndex.BUILD_BARRACKS: self.can_build_barracks
        }

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def reset(self):
        if self.env is None:
            self.init_env()

        self.supply_depot_index = 0
        self.barracks_index = 0
        self.rallies_set: Set[int] = set()

        self.raw_obs = self.env.reset()[0]
        self.player_on_left = self.get_units(Terran.CommandCenter)[0].x < 32
        self.supply_depot_locations = self.get_supply_depot_locations()
        self.barracks_locations = self.get_barracks_locations()
        self.enemy_base_location = self.get_enemy_base_location()

        return self.get_derived_obs()

    def step(self, action: Optional[np.ndarray] = None):
        self.raw_obs = self.env.step(self.get_actions(action))[0]
        derived_obs = self.get_derived_obs()
        return derived_obs, self.raw_obs.reward, self.raw_obs.last(), {}

    def get_actions(self, action: Union[np.ndarray, int]) -> List:
        mapped_actions = self.send_idle_workers_to_work()
        mapped_actions.extend(self.process_actions(action))
        mapped_actions.append(self.lower_supply_depots())

        mapped_actions = list(filter(lambda x: x is not None, mapped_actions))
        return mapped_actions

    def process_actions(self, action: Union[np.ndarray, int]) -> List:
        mapped_actions = []
        if self.is_discrete:
            for action_index, action_func in self.action_mapping.items():
                if action_index == action:
                    mapped_actions.append(action_func())
                    break
        else:
            for action_index, action_func in self.action_mapping.items():
                if action[action_index]:
                    mapped_actions.append(action_func())
        return mapped_actions

    def send_idle_workers_to_work(self) -> List:
        idle_scvs = list(filter(lambda u: u[FeatureUnit.order_length] == 0, self.get_units(Terran.SCV)))
        mineral = self.get_random_mineral()
        if mineral is None:
            return []
        orders = []
        for s_i, idle_scv in enumerate(idle_scvs):
            orders.append(actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", idle_scv[FeatureUnit.tag], mineral[FeatureUnit.tag]
            ))
        return orders

    def can_build_scv(self) -> bool:
        player = self.raw_obs.observation.player
        if player[Player.food_cap] - player[Player.food_used] < 1:
            return False
        if player[Player.minerals] < 50:
            return False
        if len(self.get_units(Terran.SCV)) >= self.scv_limit:
            return False
        ccs = self.get_units(Terran.CommandCenter)
        if len(ccs) == 0:
            return False

        cc = ccs[0]
        if cc[FeatureUnit.order_length] > 0:
            return False

        return True

    def build_scv(self):
        if self.can_build_scv():
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now", self.get_units(Terran.CommandCenter)[0].tag)
        return None

    def can_build_supply_depot(self) -> bool:
        if self.raw_obs.observation.player[Player.minerals] < 100:
            return False
        if self.supply_depot_index >= len(self.supply_depot_locations):
            return False
        return True

    def build_supply_depot(self):
        if not self.can_build_supply_depot():
            return None

        location = self.supply_depot_locations[self.supply_depot_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None

        self.supply_depot_index += 1
        return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", worker[FeatureUnit.tag], location)

    def can_build_barracks(self) -> bool:
        if self.raw_obs.observation.player[Player.minerals] < 150:
            return False
        if self.barracks_index >= len(self.barracks_locations):
            return False
        return True

    def build_barracks(self):
        if not self.can_build_barracks():
            return None

        location = self.barracks_locations[self.barracks_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None

        self.barracks_index += 1
        return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", worker[FeatureUnit.tag], location)

    def get_free_barracks(self):
        barracks = self.get_units(Terran.Barracks)
        for b in barracks:
            if b[FeatureUnit.build_progress] < 100:
                continue
            if b[FeatureUnit.order_length] > 0:
                continue
            return b
        return None

    def can_build_marine(self) -> bool:
        player = self.raw_obs.observation.player
        if player[Player.food_cap] - player[Player.food_used] < 1:
            return False
        if player[Player.minerals] < 50:
            return False
        if self.get_free_barracks() is None:
            return False
        return True

    def build_marine(self):
        if not self.can_build_marine():
            return None

        barracks = self.get_free_barracks()
        if barracks is None:
            self.logger.warning(f"Free barracks not found")
            return

        return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

    def get_random_mineral(self):
        minerals = self.get_units(Neutral.MineralField)
        if len(minerals) == 0:
            return None
        return random.choice(minerals)

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

    def get_units(self, unit_type: int, alliance: int = 1):
        return [unit for unit in self.raw_obs.observation.raw_units
                if unit.unit_type == unit_type and unit.alliance == alliance]

    def get_derived_obs(self) -> np.ndarray:
        obs = np.zeros(shape=self.observation_space.shape)
        player = self.raw_obs.observation.player
        obs[ObservationIndex.MINERALS] = player[Player.minerals] / 500
        obs[ObservationIndex.SUPPLY_TAKEN] = player[Player.food_used] / 150
        obs[ObservationIndex.SUPPLY_ALL] = player[Player.food_cap] / 150
        obs[ObservationIndex.SUPPLY_FREE] = (player[Player.food_cap] - player[Player.food_used]) / 16
        obs[ObservationIndex.SCV_COUNT] = len(self.get_units(Terran.SCV)) / self.scv_limit
        obs[ObservationIndex.TIME_STEP] = self.raw_obs.observation.game_loop / 20000
        obs[ObservationIndex.SUPPLY_DEPOT_COUNT] = self.supply_depot_index / len(self.supply_depot_locations)
        obs[ObservationIndex.IS_SUPPLY_DEPOT_BUILDING] = self.get_supply_depots_in_progress()
        obs[ObservationIndex.BARRACKS_COUNT] = self.barracks_index / len(self.barracks_locations)
        obs[ObservationIndex.IS_BARRACKS_BUILDING] = self.get_barracks_in_progress()
        obs[ObservationIndex.CAN_BUILD_MARINE] = self.can_build_marine()
        obs[ObservationIndex.CAN_BUILD_SCV] = self.can_build_scv()
        obs[ObservationIndex.CAN_BUILD_BARRACKS] = self.can_build_barracks()
        obs[ObservationIndex.CAN_BUILD_SUPPLY_DEPOT] = self.can_build_supply_depot()
        obs[ObservationIndex.SCV_IN_PROGRESS] = self.get_svc_in_progress()
        obs[ObservationIndex.MARINES_IN_PROGRESS] = self.get_marines_in_progress() / len(self.barracks_locations)
        return obs

    def get_supply_taken(self) -> int:
        return self.raw_obs.observation.player[Player.food_used]

    def get_supply_cap(self) -> int:
        return self.raw_obs.observation.player[Player.food_cap]

    def get_expected_supply_cap(self) -> int:
        return 15 + self.supply_depot_index * 8

    def get_supply_depots_in_progress(self) -> int:
        return sum(
            [supply_depot[FeatureUnit.build_progress] < 100 for supply_depot in self.get_units(Terran.SupplyDepot)]
        )

    def get_barracks_in_progress(self) -> int:
        return sum(
            [supply_depot[FeatureUnit.build_progress] < 100 for supply_depot in self.get_units(Terran.Barracks)]
        )

    def get_svc_in_progress(self) -> bool:
        ccs = self.get_units(Terran.CommandCenter)
        if len(ccs) == 0:
            return False
        return ccs[0][FeatureUnit.order_length] > 0

    def get_marines_in_progress(self) -> int:
        return sum(
            [b[FeatureUnit.order_length] > 0 for b in self.get_units(Terran.Barracks)]
        )

    def get_spots_to_build(self) -> np.ndarray:
        minimap_features = self.raw_obs.observation.feature_minimap
        spots = np.array(minimap_features.buildable - minimap_features.player_relative.astype(bool))
        return spots

    def get_supply_depot_locations(self) -> np.ndarray:
        cc = self.get_units(Terran.CommandCenter)[0]
        side_multiplier = (1 if self.player_on_left else -1)
        start_position = (cc.x + 4 * side_multiplier, cc.y)
        positions = []
        for x in range(0, 16, 2):
            for y in range(-5, 6, 5):
                positions.append([start_position[0] + x * side_multiplier, start_position[1] + y])
        positions = sorted(positions, key=lambda p: (p[0] * side_multiplier, abs(p[1])))
        return np.array(positions)

    def get_barracks_locations(self) -> np.ndarray:
        cc = self.get_units(Terran.CommandCenter)[0]
        side_multiplier = (1 if self.player_on_left else -1)
        start_position = (cc.x - 1 * side_multiplier, cc.y + 2)
        positions = []
        for x in range(0, 16, 5):
            for y in range(-10, 6, 5):
                positions.append([start_position[0] + x * side_multiplier, start_position[1] + y])
        positions = sorted(positions, key=lambda p: (p[0] * side_multiplier, abs(p[1])))
        return np.array(positions)

    def get_enemy_base_location(self) -> np.ndarray:
        return np.array(self.base_locations[-1] if self.player_on_left else self.base_locations[0])

    def attack(self):
        units = self.get_units(Terran.Marine)
        if len(units) == 0:
            return None
        tags = [u.tag for u in units]
        return actions.RAW_FUNCTIONS.Attack_pt("now", tags, self.enemy_base_location)

    def lower_supply_depots(self):
        supply_depots = self.get_units(Terran.SupplyDepot)
        valid_supply_depots = list(filter(lambda s: s.build_progress == 100 and s.order_length == 0, supply_depots))
        if len(valid_supply_depots) == 0:
            return None
        tags = [s.tag for s in valid_supply_depots]
        return actions.RAW_FUNCTIONS.Morph_SupplyDepot_Lower_quick("now", tags)

    def render(self, mode="human"):
        pass

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def action_masks(self) -> np.ndarray:

        if self.is_discrete:
            self.action_space: Discrete
            mask = [True] * self.action_space.n
            for action_index, action_func in self.valid_action_mapping.items():
                mask[action_index] = action_func()
            return np.array(mask)
        else:
            raise NotImplementedError("Action mask not implemented for non-discrete action space")
