import logging
from collections import defaultdict
from enum import IntEnum
import random
from typing import Optional, List, Set, Union

import gym
import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env, Dimensions, Difficulty
from pysc2.lib import features, actions
from pysc2.lib.features import FeatureUnit, Player
from pysc2.lib.units import Terran, Neutral, Zerg

from minigames.collect_minerals_and_gas.src.env import OrderId


class ActionIndex(IntEnum):
    BUILD_MARINE = 0
    BUILD_SCV = 1
    BUILD_SUPPLY = 2
    BUILD_BARRACKS = 3
    ATTACK = 4
    STOP_ARMY = 5
    RETREAT = 6
    GATHER_ARMY = 7
    BUILD_CC = 8

    @staticmethod
    def int_to_name(value: int):
        if value == len(ActionIndex):
            return "NO_ACTION"
        return ActionIndex(value).name


class ObservationIndex(IntEnum):
    MINERALS = 0        # scale 500
    SUPPLY_TAKEN = 1    # scale 200
    SUPPLY_ALL = 2      # scale 200
    SUPPLY_FREE = 3     # scale 16
    SCV_COUNT = 4       # scale 50
    TIME_STEP = 5       # scale 28800
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
    MILITARY_COUNT = 16     # scale 100
    CC_STARTED_BUILDING = 17
    CC_BUILT = 18


class BuildMarinesEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    rally_position = np.array([20, 36])
    map_dimensions = (88, 96)
    # no sure about the 2nd location
    base_locations = [(26, 35), (57, 31), (23, 72), (54, 68)]
    target_tags_to_ignore = {Zerg.Changeling, Zerg.ChangelingMarine, Zerg.ChangelingMarineShield,
                             Zerg.ChangelingZergling, Zerg.ChangelingZealot, Zerg.Larva, Zerg.Cocoon}
    minerals_tags = {Neutral.MineralField, Neutral.MineralField450, Neutral.MineralField750}
    cc_optimal_workers = 16
    cc_max_workers = 24
    refinery_max_workers = 3
    mineral_max_workers = 3
    mineral_optimal_workers = 2
    max_game_step = 28800
    army_actions = {ActionIndex.ATTACK, ActionIndex.RETREAT, ActionIndex.STOP_ARMY, ActionIndex.GATHER_ARMY}

    def __init__(self, step_mul: int = 8, realtime: bool = False, is_discrete: bool = True,
                 supple_depot_limit: Optional[int] = None,
                 difficulty: Difficulty = Difficulty.medium):
        self.settings = {
            'map_name': "Simple64_towers",
            'players': [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, difficulty)],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                feature_dimensions=Dimensions(screen=self.map_dimensions, minimap=self.map_dimensions),
                use_feature_units=True,
                crop_to_playable_area=True,
                show_placeholders=True,
                allow_cheating_layers=False
            ),
            'realtime': realtime,
            'step_mul': step_mul
        }

        self.is_discrete = is_discrete
        self.supple_depot_limit = supple_depot_limit
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
        self.player_on_left = False
        self.supply_depot_locations = np.zeros(shape=(0, 2))
        self.production_buildings_locations = np.zeros(shape=(0, 2))
        self.enemy_base_location = np.zeros(shape=(2,))
        self.cc_started = False
        self.cc_rally_canceled = False

        self.action_mapping = {
            ActionIndex.BUILD_MARINE: self.build_marine,
            ActionIndex.BUILD_SCV: self.build_scv,
            ActionIndex.BUILD_SUPPLY: self.build_supply_depot,
            ActionIndex.BUILD_BARRACKS: self.build_barracks,
            ActionIndex.ATTACK: self.attack,
            ActionIndex.STOP_ARMY: self.stop_army,
            ActionIndex.RETREAT: self.retreat,
            ActionIndex.GATHER_ARMY: self.gather_army,
            ActionIndex.BUILD_CC: self.build_cc
        }
        self.valid_action_mapping = {
            ActionIndex.BUILD_MARINE: self.can_build_marine,
            ActionIndex.BUILD_SCV: self.can_build_scv,
            ActionIndex.BUILD_SUPPLY: self.can_build_supply_depot,
            ActionIndex.BUILD_BARRACKS: self.can_build_barracks,
            ActionIndex.ATTACK: self.has_any_military_units,
            ActionIndex.STOP_ARMY: self.has_any_military_units,
            ActionIndex.RETREAT: self.has_any_military_units,
            ActionIndex.GATHER_ARMY: self.has_any_military_units,
            ActionIndex.BUILD_CC: self.can_build_cc
        }

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def reset(self):
        if self.env is None:
            self.init_env()

        self.supply_depot_index = 0
        self.barracks_index = 0

        self.raw_obs = self.env.reset()[0]
        self.player_on_left = self.get_units(Terran.CommandCenter, alliance=1)[0].x < 32
        self.supply_depot_locations = self.get_supply_depot_locations()
        self.production_buildings_locations = self.get_barracks_locations()
        self.enemy_base_location = self.get_enemy_base_location()
        self.cc_started = False
        self.cc_rally_canceled = False

        return self.get_derived_obs()

    def step(self, action: Optional[np.ndarray] = None):
        self.raw_obs = self.env.step(self.get_actions(action))[0]
        derived_obs = self.get_derived_obs()
        if self.should_surrender():
            return derived_obs, -1, True, {}
        return derived_obs, self.raw_obs.reward, self.raw_obs.last(), {}

    def get_actions(self, action: Union[np.ndarray, int]) -> List:
        mapped_actions = [self.cancel_cc_rally()]
        mapped_actions.extend(self.process_actions(action))
        mapped_actions.extend(self.send_idle_workers_to_work())
        mapped_actions.append(self.lower_supply_depots())

        mapped_actions = list(filter(lambda x: x is not None, mapped_actions))
        return mapped_actions

    def process_actions(self, action: Union[np.ndarray, int]) -> List:
        mapped_actions = []
        if self.is_discrete:
            for action_index, action_func in self.action_mapping.items():
                if action_index == action:
                    mapped_actions.extend(action_func())
                    break
        else:
            for action_index, action_func in self.action_mapping.items():
                if action[action_index]:
                    mapped_actions.extend(action_func())
        return mapped_actions

    def send_idle_workers_to_work(self) -> List:
        idle_scvs = list(filter(lambda u: u[FeatureUnit.order_length] == 0, self.get_units(Terran.SCV, alliance=1)))
        working_targets = self.get_working_targets(len(idle_scvs))
        orders = []
        for s_i, idle_scv in enumerate(idle_scvs):
            if s_i >= len(working_targets):
                break
            orders.append(actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", idle_scv[FeatureUnit.tag], working_targets[s_i][FeatureUnit.tag]
            ))
        return orders

    def can_build_scv(self) -> bool:
        player = self.raw_obs.observation.player
        if player[Player.food_cap] - player[Player.food_used] < 1:
            return False
        if player[Player.minerals] < 50:
            return False
        if len(self.get_units(Terran.SCV, alliance=1)) >= self.cc_max_workers * (1 + self.is_2nd_cc_built()):
            return False
        ccs = self.get_units(Terran.CommandCenter, alliance=1)
        if len(ccs) == 0:
            return False

        if all(cc[FeatureUnit.order_length] > 0 for cc in ccs):
            return False

        return True

    def build_scv(self) -> List:
        if self.can_build_scv():
            ccs = self.get_units(Terran.CommandCenter, alliance=1)
            for cc in ccs:
                if cc[FeatureUnit.order_length] == 0:
                    return [actions.RAW_FUNCTIONS.Train_SCV_quick("now", cc.tag)]
            else:
                self.logger.warning("No free CC to build an scv")
        return []

    def can_build_supply_depot(self) -> bool:
        if self.raw_obs.observation.player[Player.minerals] < 100:
            return False
        if self.supply_depot_index >= len(self.supply_depot_locations):
            return False
        if self.supple_depot_limit is not None and \
                len(self.get_units({Terran.SupplyDepot, Terran.SupplyDepotLowered}, alliance=1)) \
                >= self.supple_depot_limit:
            return False
        return True

    def build_supply_depot(self) -> List:
        if not self.can_build_supply_depot():
            return []

        location = self.supply_depot_locations[self.supply_depot_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return []

        self.supply_depot_index += 1
        return [actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", worker[FeatureUnit.tag], location)]

    def can_build_barracks(self) -> bool:
        if self.raw_obs.observation.player[Player.minerals] < 150:
            return False
        if self.barracks_index >= len(self.production_buildings_locations):
            return False
        return True

    def build_barracks(self) -> List:
        if not self.can_build_barracks():
            return []

        location = self.production_buildings_locations[self.barracks_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return []

        self.barracks_index += 1
        return [actions.RAW_FUNCTIONS.Build_Barracks_pt("now", worker[FeatureUnit.tag], location)]

    def get_free_barracks(self):
        barracks = self.get_units(Terran.Barracks, alliance=1)
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

    def build_marine(self) -> List:
        if not self.can_build_marine():
            return []

        barracks = self.get_free_barracks()
        if barracks is None:
            self.logger.warning(f"Free barracks not found")
            return []

        return [actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)]

    def get_random_mineral(self):
        minerals = self.get_units(self.minerals_tags)
        if len(minerals) == 0:
            return None
        return random.choice(minerals)

    def get_working_targets(self, n_idle_workers: int) -> List:
        minerals = self.get_units(self.minerals_tags)
        refineries = list(filter(lambda r: r[FeatureUnit.build_progress] == 100,
                                 self.get_units(Terran.Refinery, alliance=1)))
        ccs = sorted(list(
            filter(lambda c: c[FeatureUnit.build_progress] == 100, self.get_units(Terran.CommandCenter, alliance=1))
        ), key=lambda c: c[FeatureUnit.x])
        cc_to_minerals = defaultdict(list)
        for mineral in minerals:
            for cc_idx, cc in enumerate(ccs):
                if np.power(mineral.x - cc.x, 2) + np.power(mineral.y - cc.y, 2) < 100:
                    cc_to_minerals[cc_idx].append(mineral)

        cc_allocations = [c[FeatureUnit.assigned_harvesters] for c in ccs]
        refinery_allocations = [r[FeatureUnit.assigned_harvesters] for r in refineries]
        worker_targets: List = []
        for w_i in range(n_idle_workers):
            for c_i in range(len(ccs)):
                if cc_allocations[c_i] < len(cc_to_minerals[c_i]) * self.mineral_optimal_workers:
                    cc_allocations[c_i] += 1
                    worker_targets.append(random.choice(cc_to_minerals[c_i]))
                    break
            for r_i in range(len(refineries)):
                if refinery_allocations[r_i] < self.refinery_max_workers:
                    refinery_allocations[r_i] += 1
                    worker_targets.append(refineries[r_i])
                    break
            for c_i in range(len(ccs)):
                if cc_allocations[c_i] < len(cc_to_minerals[c_i]) * self.mineral_max_workers:
                    cc_allocations[c_i] += 1
                    worker_targets.append(random.choice(cc_to_minerals[c_i]))
                    break
        return worker_targets

    def get_nearest_worker(self, location: np.ndarray):
        all_workers = self.get_units(Terran.SCV, alliance=1)
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

    def get_units(self, unit_type: Optional[Union[int, Set[int]]] = None,
                  alliance: Optional[Union[int, Set[int]]] = None) -> List:
        units = self.raw_obs.observation.raw_units
        if unit_type is not None:
            if isinstance(unit_type, int):
                unit_type = {unit_type}
            units = list(filter(lambda u: u.unit_type in unit_type, units))
        if alliance is not None:
            if isinstance(alliance, int):
                alliance = {alliance}
            units = list(filter(lambda u: u.alliance in alliance, units))
        return units

    def get_derived_obs(self) -> np.ndarray:
        obs = np.zeros(shape=self.observation_space.shape)
        player = self.raw_obs.observation.player
        obs[ObservationIndex.MINERALS] = player[Player.minerals] / 500
        obs[ObservationIndex.SUPPLY_TAKEN] = player[Player.food_used] / 200
        obs[ObservationIndex.SUPPLY_ALL] = player[Player.food_cap] / 200
        obs[ObservationIndex.SUPPLY_FREE] = (player[Player.food_cap] - player[Player.food_used]) / 16
        obs[ObservationIndex.SCV_COUNT] = len(self.get_units(Terran.SCV, alliance=1)) / 50
        obs[ObservationIndex.TIME_STEP] = self.get_normalized_time()
        obs[ObservationIndex.SUPPLY_DEPOT_COUNT] = self.supply_depot_index / len(self.supply_depot_locations)
        obs[ObservationIndex.IS_SUPPLY_DEPOT_BUILDING] = self.get_supply_depots_in_progress()
        obs[ObservationIndex.BARRACKS_COUNT] = self.barracks_index / len(self.production_buildings_locations)
        obs[ObservationIndex.IS_BARRACKS_BUILDING] = self.get_barracks_in_progress()
        obs[ObservationIndex.CAN_BUILD_MARINE] = self.can_build_marine()
        obs[ObservationIndex.CAN_BUILD_SCV] = self.can_build_scv()
        obs[ObservationIndex.CAN_BUILD_BARRACKS] = self.can_build_barracks()
        obs[ObservationIndex.CAN_BUILD_SUPPLY_DEPOT] = self.can_build_supply_depot()
        obs[ObservationIndex.SCV_IN_PROGRESS] = self.get_svc_in_progress()
        obs[ObservationIndex.MARINES_IN_PROGRESS] = \
            self.get_marines_in_progress() / len(self.production_buildings_locations)
        obs[ObservationIndex.MILITARY_COUNT] = player[Player.army_count] / 100
        obs[ObservationIndex.CC_STARTED_BUILDING] = float(self.cc_started)
        obs[ObservationIndex.CC_BUILT] = float(self.is_2nd_cc_built())
        return obs

    def get_supply_taken(self) -> int:
        return self.raw_obs.observation.player[Player.food_used]

    def get_supply_cap(self) -> int:
        return self.raw_obs.observation.player[Player.food_cap]

    def get_expected_supply_cap(self) -> int:
        return 15 + self.supply_depot_index * 8

    def get_supply_depots_in_progress(self) -> int:
        return sum(
            [supply_depot[FeatureUnit.build_progress] < 100 for supply_depot
             in self.get_units(Terran.SupplyDepot, alliance=1)]
        )

    def get_barracks_in_progress(self) -> int:
        return sum(
            [supply_depot[FeatureUnit.build_progress] < 100 for supply_depot
             in self.get_units(Terran.Barracks, alliance=1)]
        )

    def get_svc_in_progress(self) -> bool:
        ccs = self.get_units(Terran.CommandCenter, alliance=1)
        if len(ccs) == 0:
            return False
        return ccs[0][FeatureUnit.order_length] > 0

    def get_marines_in_progress(self) -> int:
        return sum(
            [b[FeatureUnit.order_length] > 0 for b in self.get_units(Terran.Barracks, alliance=1)]
        )

    def get_spots_to_build(self) -> np.ndarray:
        minimap_features = self.raw_obs.observation.feature_minimap
        spots = np.array(minimap_features.buildable - minimap_features.player_relative.astype(bool))
        return spots

    def get_supply_depot_locations(self) -> np.ndarray:
        cc = self.get_units(Terran.CommandCenter, alliance=1)[0]
        side_multiplier = (1 if self.player_on_left else -1)
        start_position = (cc.x + 4 * side_multiplier, cc.y)
        positions = []
        for x in range(0, 16, 2):
            for y in range(-5, 6, 5):
                positions.append([start_position[0] + x * side_multiplier, start_position[1] + y])
        positions = sorted(positions, key=lambda p: (p[0] * side_multiplier, abs(p[1])))
        return np.array(positions)

    def get_barracks_locations(self) -> np.ndarray:
        cc = self.get_units(Terran.CommandCenter, alliance=1)[0]
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

    def get_military_units(self) -> List:
        return self.get_units({Terran.Marine}, alliance=1)

    def has_any_military_units(self) -> bool:
        return len(self.get_military_units()) > 0

    def attack_enemy_base_location(self):
        units = self.get_military_units()
        if len(units) == 0:
            return None
        tags = [u.tag for u in units]
        self.logger.debug(f"Attack enemy base location: {self.enemy_base_location}")
        return actions.RAW_FUNCTIONS.Attack_pt("now", tags, self.enemy_base_location)

    def attack_nearest_target(self) -> List:
        targets = self.get_units(alliance=4)
        targets = list(filter(lambda t: t.unit_type not in self.target_tags_to_ignore, targets))
        if len(targets) == 0:
            return []
        units = self.get_military_units()
        if len(units) == 0:
            return []

        unit_positions = np.array([[u.x, u.y] for u in units])
        target_positions = np.array([[t.x, t.y] for t in targets])
        distances = np.zeros(shape=(len(unit_positions), len(target_positions)))
        for u_idx, u_pos in enumerate(unit_positions):
            for t_idx, t_pos in enumerate(target_positions):
                distances[u_idx, t_idx] = np.sum(np.power(u_pos - t_pos, 2))
        unit_to_target_mapping = distances.argmin(axis=1)
        target_to_units = defaultdict(set)
        for u_idx, t_dx in enumerate(unit_to_target_mapping):
            target_to_units[t_dx].add(u_idx)

        attack_list = []
        for t_dx, unit_indices in target_to_units.items():
            target_location = np.array([targets[t_dx].x, targets[t_dx].y])
            tags = [u.tag for u_idx, u in enumerate(units) if u_idx in unit_indices]
            attack_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", tags, target_location))
            self.logger.debug(f"Attack nearest target: {target_location}, tags: {tags}")
        return attack_list

    def attack(self) -> List:
        return self.attack_nearest_target() or [self.attack_enemy_base_location()]

    def stop_army(self) -> List:
        units = self.get_military_units()
        if len(units) == 0:
            return []
        tags = [u.tag for u in units]
        return [actions.RAW_FUNCTIONS.Stop_quick("now", tags)]

    def get_retreat_position(self) -> np.ndarray:
        if self.player_on_left:
            return np.array(self.base_locations[0]) + np.array([20., 0.])
        else:
            return np.array(self.base_locations[-1]) + np.array([-20., 0.])

    def get_center_of_military_units(self) -> np.ndarray:
        units = self.get_military_units()
        if len(units) == 0:
            return np.array([0., 0.])
        positions = np.array([[u.x, u.y] for u in units])
        return np.median(positions, axis=0)

    def gather_army(self) -> List:
        units = self.get_military_units()
        if len(units) == 0:
            return []
        tags = [u.tag for u in units]
        location = self.get_center_of_military_units()
        self.logger.debug(f"Gather army location: {location}")
        return [actions.RAW_FUNCTIONS.Attack_pt("now", tags, location)]

    def retreat(self) -> List:
        units = self.get_military_units()
        if len(units) == 0:
            return []
        tags = [u.tag for u in units]
        return [actions.RAW_FUNCTIONS.Move_pt("now", tags, self.get_retreat_position())]

    def lower_supply_depots(self):
        supply_depots = self.get_units(Terran.SupplyDepot, alliance=1)
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

    def get_score_cumulative(self):
        return self.raw_obs.observation.score_cumulative

    def get_score_by_category(self):
        return self.raw_obs.observation.score_by_category

    def should_surrender(self) -> bool:
        return len(self.get_units(Terran.CommandCenter, alliance=1)) == 0

    def get_normalized_time(self) -> float:
        return self.raw_obs.observation.game_loop[0] / self.max_game_step

    def is_2nd_cc_built(self) -> bool:
        ccs = sorted(self.get_units(Terran.CommandCenter, alliance=1), key=lambda c: c[FeatureUnit.x])
        return len(ccs) == 2 and ccs[1][FeatureUnit.build_progress] == 100

    def can_build_cc(self) -> bool:
        player = self.raw_obs.observation.player
        return player.minerals >= 400 and not self.cc_started

    def build_cc(self) -> List:
        if not self.can_build_cc():
            return []
        location = np.array(self.base_locations[1 if self.player_on_left else 2])
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return []
        self.cc_started = True
        return [actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", worker[FeatureUnit.tag], location)]

    def cancel_cc_rally(self):
        if self.cc_rally_canceled:
            return None
        cc = self.get_units(Terran.CommandCenter, alliance=1)[0]
        location = (cc[FeatureUnit.x], cc[FeatureUnit.y])
        self.cc_rally_canceled = True
        return actions.RAW_FUNCTIONS.Rally_Workers_pt("now", cc[FeatureUnit.tag], location)
