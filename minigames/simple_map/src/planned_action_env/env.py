import functools
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Set, Union, Dict, Tuple

import gym
import numpy as np
from gym.spaces import Box, Discrete
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env, Dimensions, Difficulty, Race
from pysc2.lib import features, actions
from pysc2.lib.features import FeatureUnit, Player, PlayerRelative
from pysc2.lib.units import Terran, Neutral, Zerg, Protoss
from pysc2.lib.upgrades import Upgrades

from common.building_requirements import get_building_requirement
from common.unit_cost import unit_to_cost, Cost, get_item_cost, upgrade_to_cost
from minigames.collect_minerals_and_gas.src.env import OrderId
from minigames.simple_map.src.planned_action_env.difficulty_scheduler import DifficultyScheduler
from minigames.simple_map.src.planned_action_env.reward_shaper import RewardShaper


class ActionIndex(IntEnum):
    BUILD_MARINE = 0
    BUILD_SCV = 1
    BUILD_SUPPLY = 2
    BUILD_BARRACKS = 3
    BUILD_CC = 4
    ATTACK = 5
    STOP_ARMY = 6
    RETREAT = 7
    GATHER_ARMY = 8
    NOTHING = 9
    BUILD_REFINERY = 10
    BUILD_ENGINEERING_BAY = 11
    RESEARCH_INFANTRY_WEAPONS = 12
    BUILD_FACTORY = 13
    BUILD_ARMORY = 14
    RESEARCH_INFANTRY_ARMOR = 15


class ObservationIndex(IntEnum):
    MINERALS = 0  # scale 500
    SUPPLY_TAKEN = 1  # scale 200
    SUPPLY_ALL = 2  # scale 200
    SUPPLY_FREE = 3  # scale 16
    SCV_COUNT = 4  # scale 50
    TIME_STEP = 5  # scale 28800
    SUPPLY_DEPOT_COUNT = 6
    IS_SUPPLY_DEPOT_BUILDING = 7
    PRODUCTION_BUILDING_COUNT = 8
    IS_BARRACKS_BUILDING = 9
    MILITARY_COUNT = 10  # scale 100
    CC_STARTED_BUILDING = 11
    CC_BUILT = 12
    ENEMY_RACE_TERRAN = 13
    ENEMY_RACE_ZERG = 14
    ENEMY_RACE_PROTOSS = 15
    ENEMY_PROXIMITY = 16
    REFINERY_COUNT = 17  # scale max_refinery
    IS_REFINERY_BUILDING = 18
    ENGINEERING_BAY_COUNT = 19  # scale 2
    IS_ENGINEERING_BAY_BUILDING = 20
    INFANTRY_WEAPONS_COMPLETED = 21  # scale 3
    ARMORY_COUNT = 22
    IS_ARMORY_BUILDING = 23
    FACTORY_COUNT = 24
    IS_FACTORY_BUILDING = 25
    INFANTRY_ARMOR_COMPLETED = 26  # scale 3


supply_limit = 200
cc_optimal_workers = 16
cc_max_workers = 24
refinery_max_workers = 3
mineral_max_workers = 3
mineral_optimal_workers = 2
building_limits = {
    Terran.EngineeringBay: 2,
    Terran.Armory: 1,
    Terran.Factory: 1
}

action_to_unit = {
    ActionIndex.BUILD_MARINE: Terran.Marine,
    ActionIndex.BUILD_CC: Terran.CommandCenter,
    ActionIndex.BUILD_SCV: Terran.SCV,
    ActionIndex.BUILD_SUPPLY: Terran.SupplyDepot,
    ActionIndex.BUILD_BARRACKS: Terran.Barracks,
    ActionIndex.BUILD_REFINERY: Terran.Refinery,
    ActionIndex.BUILD_ENGINEERING_BAY: Terran.EngineeringBay
}

action_to_upgrade = {
    ActionIndex.RESEARCH_INFANTRY_WEAPONS: {Upgrades.TerranInfantryWeaponsLevel1, Upgrades.TerranInfantryWeaponsLevel2,
                                            Upgrades.TerranInfantryWeaponsLevel3},
    ActionIndex.RESEARCH_INFANTRY_ARMOR: {Upgrades.TerranInfantryArmorsLevel1, Upgrades.TerranInfantryArmorsLevel2,
                                          Upgrades.TerranInfantryArmorsLevel3}
}


@dataclass
class ActionRequirement:
    minerals: bool = False
    vespene: bool = False
    buildings: List[int] = field(default_factory=lambda: [])
    queue: bool = False
    invalid: bool = False

    @property
    def can_do_instantly(self) -> bool:
        return not self.minerals and not self.vespene and len(self.buildings) == 0 and not self.queue \
               and not self.invalid

    def to_numpy(self) -> np.ndarray:
        return np.array([self.minerals, self.vespene, len(self.buildings), self.queue, self.invalid,
                         self.can_do_instantly])


class BuildingTypeState(IntEnum):
    NOT_PRESENT = 0
    IS_BUILDING = 1
    IS_BUILT = 2

    def to_numpy(self) -> np.ndarray:
        return np.identity(len(self.__class__))[self.value]


class PlannedActionEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    map_dimensions = (88, 96)
    base_locations = [(26, 35), (57, 31), (23, 72), (54, 68)]
    attack_locations = [
        [(50, 45), (38, 48), (27, 49), (23, 58), (23, 72), (54, 68)],
        [(30, 58), (42, 55), (53, 54), (57, 45), (57, 31), (26, 35)]
    ]
    upgrade_building_locations = [
        [(16, 38), (16, 34), (16, 30), (19, 26), (23, 26), (26, 26)],
        [(63, 65), (63, 69), (63, 72), (60, 77), (58, 77), (54, 77)]
    ]
    target_tags_to_ignore = {Zerg.Changeling, Zerg.ChangelingMarine, Zerg.ChangelingMarineShield,
                             Zerg.ChangelingZergling, Zerg.ChangelingZealot, Zerg.Larva, Zerg.Cocoon, Zerg.Overseer,
                             Zerg.Overlord, Zerg.OverlordTransport, Zerg.OverseerCocoon, Zerg.OverseerOversightMode}
    minerals_tags = {Neutral.MineralField, Neutral.MineralField450, Neutral.MineralField750}

    max_game_step = 28800
    army_actions = {ActionIndex.ATTACK, ActionIndex.RETREAT, ActionIndex.STOP_ARMY, ActionIndex.GATHER_ARMY}
    building_types = [Terran.SupplyDepot, Terran.Barracks, Terran.Refinery, Terran.EngineeringBay, Terran.Factory,
                      Terran.Armory]
    production_building_types = {Terran.Barracks}

    def __init__(self, step_mul: int = 8, realtime: bool = False, difficulty: Difficulty = Difficulty.medium,
                 enemy_race: sc2_env.Race = sc2_env.Race.random,
                 reward_shapers: Optional[List[RewardShaper]] = None,
                 time_to_finishing_move: float = 0.8, supply_to_finishing_move: int = 200,
                 free_supply_margin_factor: float = 2.0, output_path: Optional[str] = None,
                 difficulty_scheduler: Optional[DifficultyScheduler] = None,
                 max_refineries: int = 2):
        self.settings = {
            'map_name': "Simple64_towers",
            'players': [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(enemy_race, difficulty)],
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
        self.action_space = Discrete(len(ActionIndex))
        self.observation_space = Box(low=0.0, high=1.0, shape=(
            len(ObservationIndex) + len(ActionIndex) * len(ActionRequirement().to_numpy())
            + len(self.building_types) * len(BuildingTypeState(0).to_numpy()),
        ))
        self.env: Optional[SC2Env] = None
        self.logger = logging.getLogger("PlannedActionEnv")
        self.raw_obs = None
        self.reward_shapers: List[RewardShaper] = reward_shapers or []
        for reward_shaper in self.reward_shapers:
            reward_shaper.set_env(self)
        self.time_to_finishing_move = time_to_finishing_move
        self.supply_to_finishing_move = supply_to_finishing_move
        self.free_supply_margin_factor = free_supply_margin_factor
        self.output_path = output_path

        self.supply_depot_index = 0
        self.production_building_index = 0
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
            ActionIndex.BUILD_CC: self.build_cc,
            ActionIndex.NOTHING: self.do_nothing,
            ActionIndex.BUILD_REFINERY: self.build_refinery,
            ActionIndex.BUILD_ENGINEERING_BAY: self.build_engineering_bay,
            ActionIndex.RESEARCH_INFANTRY_WEAPONS: self.research_infantry_weapons,
            ActionIndex.BUILD_FACTORY: self.build_factory,
            ActionIndex.BUILD_ARMORY: self.build_armory,
            ActionIndex.RESEARCH_INFANTRY_ARMOR: self.research_infantry_armor,
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
            ActionIndex.BUILD_CC: self.can_build_cc,
            ActionIndex.NOTHING: self.can_do_nothing,
            ActionIndex.BUILD_REFINERY: self.can_build_refinery,
            ActionIndex.BUILD_ENGINEERING_BAY: self.can_build_engineering_bay,
            ActionIndex.RESEARCH_INFANTRY_WEAPONS: self.can_research_infantry_weapons,
            ActionIndex.BUILD_FACTORY: self.can_build_factory,
            ActionIndex.BUILD_ARMORY: self.can_build_armory,
            ActionIndex.RESEARCH_INFANTRY_ARMOR: self.can_research_infantry_armor
        }
        self.building_to_action_mapping = {
            Terran.SupplyDepot: ActionIndex.BUILD_SUPPLY,
            Terran.Barracks: ActionIndex.BUILD_BARRACKS,
            Terran.Refinery: ActionIndex.BUILD_REFINERY,
            Terran.EngineeringBay: ActionIndex.BUILD_ENGINEERING_BAY,
            Terran.Armory: ActionIndex.BUILD_ARMORY,
            Terran.Factory: ActionIndex.BUILD_FACTORY
        }

        self.pending_actions: List[ActionIndex] = []
        self.building_states: Dict[Terran, BuildingTypeState] = {}
        self.action_requirements: Dict[ActionIndex, ActionRequirement] = {}
        self.last_game_step = 0
        self.rl_step = 0
        self.episode_rewards: List[float] = []
        self.episode_reward = 0.0
        self.last_action_index: ActionIndex = ActionIndex.NOTHING
        self.enemy_race = Race.random
        self.reached_enemy_main = False
        self.override_chosen_action: Optional[ActionIndex] = None
        self.difficulty_scheduler = difficulty_scheduler
        if self.difficulty_scheduler:
            self.difficulty_scheduler.current_difficulty = difficulty
        self.finishing_move_triggered = False
        self.max_refineries = max_refineries
        self.refinery_index = 0
        self.geysers = []
        self.upgrade_building_index = 0
        self.engineering_bay_index = 0
        self.armory_index = 0

    def init_env(self):
        self.env = sc2_env.SC2Env(**self.settings)

    def reset(self):
        if self.env is None:
            self.init_env()

        self.supply_depot_index = 0
        self.production_building_index = 0

        self.raw_obs = self.env.reset()[0]
        self.player_on_left = self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF)[0].x < 32
        self.supply_depot_locations = self.get_supply_depot_locations()
        self.production_buildings_locations = self.get_barracks_locations()
        self.enemy_base_location = self.get_enemy_base_location()
        self.cc_started = False
        self.cc_rally_canceled = False
        self.pending_actions: List[ActionIndex] = []
        self.update_states()
        self.last_game_step = self.get_game_step()
        self.rl_step = 0
        self.episode_reward = 0.0
        for reward_shaper in self.reward_shapers:
            reward_shaper.reset()
        self.last_action_index: ActionIndex = ActionIndex.NOTHING
        self.enemy_race = Race.random
        self.reached_enemy_main = False
        self.override_chosen_action = None
        self.finishing_move_triggered = False
        self.refinery_index = 0
        self.geysers = self.get_geysers()
        self.upgrade_building_index = 0
        self.engineering_bay_index = 0
        self.armory_index = 0

        return self.get_derived_obs()

    def step(self, action: int):
        self.rl_step += 1
        self.last_action_index = ActionIndex(action)
        if self.override_chosen_action is not None \
                and not self.valid_action_mapping[self.override_chosen_action]().invalid:
            self.last_action_index = self.override_chosen_action
        self.logger.debug(f"Chosen action: {self.last_action_index.name}, pending actions: {self.pending_actions}")

        engine_action = self.get_action() or self.get_idle_action()
        normalized_engine_action = self.normalize_engine_action(engine_action)
        self.logger.debug(f"Engine action: {normalized_engine_action}")
        self.raw_obs = self.env.step(normalized_engine_action)[0]
        self.update_states()
        shaped_rewards = self.get_shaped_rewards()

        if self.should_surrender():
            self.register_episode_reward(-1)
            return self.get_derived_obs(), -1 + shaped_rewards, True, {}
        if self.raw_obs.last():
            self.register_episode_reward(self.raw_obs.reward)
        return self.get_derived_obs(), self.raw_obs.reward + shaped_rewards, self.raw_obs.last(), {}

    def update_states(self):
        self.building_states = self.get_building_type_states()
        self.action_requirements = self.get_action_requirements()
        self.update_enemy_race()

    def normalize_engine_action(self, engine_action):
        if engine_action is None:
            return []
        if not isinstance(engine_action, list):
            return [engine_action]
        engine_action = list(filter(lambda x: x is not None, engine_action))
        return engine_action

    def get_action(self):
        if self.should_do_finishing_attack() and self.last_action_index in self.army_actions:
            self.last_action_index = ActionIndex.ATTACK

        action_requirement = self.action_requirements[self.last_action_index]

        if self.pending_actions:
            pending_action = self.pending_actions[0]
            pending_action_requirement = self.action_requirements[pending_action]
            if pending_action_requirement.can_do_instantly:
                self.pending_actions = self.pending_actions[1:]
                return self.action_mapping[pending_action]()
            if pending_action_requirement.invalid:
                self.pending_actions = self.pending_actions[1:]
            elif pending_action_requirement.buildings:
                self.pending_actions = [
                                           self.building_to_action_mapping[Terran(building)]
                                           for building in pending_action_requirement.buildings
                                       ] + self.pending_actions

        if action_requirement.can_do_instantly:
            return self.action_mapping[self.last_action_index]()

        if not self.pending_actions:
            self.pending_actions.append(self.last_action_index)
            if action_requirement.buildings:
                self.pending_actions = [
                                           self.building_to_action_mapping[Terran(building)] for building in
                                           action_requirement.buildings
                                       ] + self.pending_actions

        return None

    def get_idle_action(self):
        if self.should_do_finishing_attack():
            return self.attack()
        return self.cancel_cc_rally() or self.send_idle_worker_to_work() or self.lower_supply_depots()

    def send_idle_worker_to_work(self):
        idle_scvs = list(filter(lambda u: u[FeatureUnit.order_length] == 0,
                                self.get_units(Terran.SCV, alliance=PlayerRelative.SELF)))
        working_targets = self.get_working_targets(len(idle_scvs))
        for s_i, idle_scv in enumerate(idle_scvs):
            if s_i >= len(working_targets):
                break
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", idle_scv[FeatureUnit.tag], working_targets[s_i][FeatureUnit.tag]
            )
        return None

    def can_build_scv(self) -> ActionRequirement:
        action_requirement = self.get_requirements(Terran.SCV)
        if len(self.get_units(Terran.SCV, alliance=PlayerRelative.SELF)) \
                >= cc_max_workers * (1 + self.is_2nd_cc_built()):
            action_requirement.invalid = True
        return action_requirement

    def build_scv(self):
        ccs = self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF)
        for cc in ccs:
            if cc[FeatureUnit.order_length] == 0:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", cc.tag)
        else:
            self.logger.warning("No free CC to build an scv")
        return None

    def can_build_supply_depot(self) -> ActionRequirement:
        action_requirement = self.get_requirements(Terran.SupplyDepot)
        if self.supply_depot_index >= len(self.supply_depot_locations):
            action_requirement.invalid = True
        elif self.building_states[Terran.SupplyDepot] != BuildingTypeState.NOT_PRESENT \
                and self.get_expected_supply_cap() > self.get_supply_taken() \
                + self.free_supply_margin_factor * (1 + self.get_production_building_count()):
            action_requirement.invalid = True
        return action_requirement

    def build_supply_depot(self):
        location = self.supply_depot_locations[self.supply_depot_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None

        self.supply_depot_index += 1
        return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", worker[FeatureUnit.tag], location)

    def can_build_barracks(self) -> ActionRequirement:
        action_requirement = self.get_requirements(Terran.Barracks)
        if self.production_building_index >= len(self.production_buildings_locations):
            action_requirement.invalid = True
        return action_requirement

    def build_barracks(self):
        location = self.production_buildings_locations[self.production_building_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return []

        self.production_building_index += 1
        return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", worker[FeatureUnit.tag], location)

    def get_free_building(self, building: Optional[Union[int, Set[int]]]):
        buildings = self.get_units(building, alliance=PlayerRelative.SELF)
        for b in buildings:
            if b[FeatureUnit.build_progress] < 100:
                continue
            if b[FeatureUnit.order_length] > 0:
                continue
            return b
        return None

    def can_build_marine(self) -> ActionRequirement:
        return self.get_requirements(Terran.Marine)

    def build_marine(self):
        barracks = self.get_free_building(Terran.Barracks)
        if barracks is None:
            self.logger.warning(f"Free barracks not found")
            return None

        return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

    def get_random_mineral(self):
        minerals = self.get_units(self.minerals_tags)
        if len(minerals) == 0:
            return None
        return random.choice(minerals)

    def get_working_targets(self, n_idle_workers: int) -> List:
        minerals = self.get_units(self.minerals_tags)
        refineries = list(filter(lambda r: r[FeatureUnit.build_progress] == 100,
                                 self.get_units(Terran.Refinery, alliance=PlayerRelative.SELF)))
        ccs = sorted(list(
            filter(lambda c: c[FeatureUnit.build_progress] == 100,
                   self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF))
        ), key=lambda c: c[FeatureUnit.x if self.player_on_left else -FeatureUnit.x])
        cc_to_minerals = defaultdict(list)
        for mineral in minerals:
            for cc_idx, cc in enumerate(ccs):
                if np.power(mineral.x - cc.x, 2) + np.power(mineral.y - cc.y, 2) < 100:
                    cc_to_minerals[cc_idx].append(mineral)

        cc_allocations = [c[FeatureUnit.assigned_harvesters] for c in ccs]
        refinery_allocations = [r[FeatureUnit.assigned_harvesters] for r in refineries]
        worker_targets: List = []
        for w_i in range(n_idle_workers):
            for r_i in range(len(refineries)):
                if refinery_allocations[r_i] < refinery_max_workers:
                    refinery_allocations[r_i] += 1
                    worker_targets.append(refineries[r_i])
                    break
            for c_i in range(len(ccs)):
                if cc_allocations[c_i] < len(cc_to_minerals[c_i]) * mineral_optimal_workers:
                    cc_allocations[c_i] += 1
                    worker_targets.append(random.choice(cc_to_minerals[c_i]))
                    break
            for c_i in range(len(ccs)):
                if cc_allocations[c_i] < len(cc_to_minerals[c_i]) * mineral_max_workers:
                    cc_allocations[c_i] += 1
                    worker_targets.append(random.choice(cc_to_minerals[c_i]))
                    break
        return worker_targets

    def get_nearest_worker(self, location: Union[np.ndarray, Tuple[int, int]]):
        all_workers = self.get_units(Terran.SCV, alliance=PlayerRelative.SELF)
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
        obs[ObservationIndex.SCV_COUNT] = len(self.get_units(Terran.SCV, alliance=PlayerRelative.SELF)) / 50
        obs[ObservationIndex.TIME_STEP] = self.get_normalized_time()
        obs[ObservationIndex.SUPPLY_DEPOT_COUNT] = self.supply_depot_index / len(self.supply_depot_locations)
        obs[ObservationIndex.IS_SUPPLY_DEPOT_BUILDING] = self.get_supply_depots_in_progress()
        obs[ObservationIndex.PRODUCTION_BUILDING_COUNT] = \
            self.production_building_index / len(self.production_buildings_locations)
        obs[ObservationIndex.IS_BARRACKS_BUILDING] = self.get_barracks_in_progress()
        obs[ObservationIndex.MILITARY_COUNT] = player[Player.army_count] / 100
        obs[ObservationIndex.CC_STARTED_BUILDING] = float(self.cc_started)
        obs[ObservationIndex.CC_BUILT] = float(self.is_2nd_cc_built())
        obs[ObservationIndex.ENEMY_RACE_TERRAN] = float(self.enemy_race == Race.terran)
        obs[ObservationIndex.ENEMY_RACE_ZERG] = float(self.enemy_race == Race.zerg)
        obs[ObservationIndex.ENEMY_RACE_PROTOSS] = float(self.enemy_race == Race.protoss)
        obs[ObservationIndex.ENEMY_PROXIMITY] = self.get_enemy_proximity()
        obs[ObservationIndex.REFINERY_COUNT] = self.refinery_index / self.max_refineries
        obs[ObservationIndex.IS_REFINERY_BUILDING] = float(sum(
            [refinery[FeatureUnit.build_progress] < 100 for refinery in self.get_units(Terran.Refinery)]
        ))
        obs[ObservationIndex.ENGINEERING_BAY_COUNT] = \
            self.engineering_bay_index / building_limits[Terran.EngineeringBay]
        obs[ObservationIndex.IS_ENGINEERING_BAY_BUILDING] = float(sum(
            [b[FeatureUnit.build_progress] < 100 for b in self.get_units(Terran.EngineeringBay)]
        ))
        obs[ObservationIndex.INFANTRY_WEAPONS_COMPLETED] \
            = (float(Upgrades.TerranInfantryWeaponsLevel1 in self.raw_obs.observation.upgrades)
               + float(Upgrades.TerranInfantryWeaponsLevel2 in self.raw_obs.observation.upgrades)
               + float(Upgrades.TerranInfantryWeaponsLevel3 in self.raw_obs.observation.upgrades)) / 3.0
        obs[ObservationIndex.INFANTRY_ARMOR_COMPLETED] \
            = (float(Upgrades.TerranInfantryArmorsLevel1 in self.raw_obs.observation.upgrades)
               + float(Upgrades.TerranInfantryArmorsLevel2 in self.raw_obs.observation.upgrades)
               + float(Upgrades.TerranInfantryArmorsLevel3 in self.raw_obs.observation.upgrades)) / 3.0
        obs[ObservationIndex.ARMORY_COUNT] = \
            self.armory_index / building_limits[Terran.Armory]
        obs[ObservationIndex.IS_ARMORY_BUILDING] = float(sum(
            [b[FeatureUnit.build_progress] < 100 for b in self.get_units(Terran.Armory)]
        ))
        obs[ObservationIndex.FACTORY_COUNT] = \
            len(self.get_units(Terran.Factory)) / building_limits[Terran.Factory]
        obs[ObservationIndex.IS_FACTORY_BUILDING] = float(sum(
            [b[FeatureUnit.build_progress] < 100 for b in self.get_units(Terran.Factory)]
        ))

        obs_index = len(ObservationIndex)
        action_req_len = len(ActionRequirement().to_numpy())
        for action_index in range(len(ActionIndex)):
            obs[obs_index: obs_index + action_req_len] = \
                self.action_requirements[ActionIndex(action_index)].to_numpy()
            obs_index += action_req_len

        building_state_len = len(BuildingTypeState(0).to_numpy())
        for building_state_index in range(len(self.building_types)):
            obs[obs_index: obs_index + building_state_len] = \
                self.building_states[self.building_types[building_state_index]].to_numpy()
            obs_index += building_state_len

        return obs

    def get_supply_taken(self) -> int:
        return self.raw_obs.observation.player[Player.food_used]

    def get_supply_cap(self) -> int:
        return self.raw_obs.observation.player[Player.food_cap]

    def get_expected_supply_cap(self) -> int:
        return len(self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF)) * 15 \
               + len(self.get_units({Terran.SupplyDepot, Terran.SupplyDepotLowered}, alliance=PlayerRelative.SELF)) * 8

    def get_supply_depots_in_progress(self) -> int:
        return sum(
            [supply_depot[FeatureUnit.build_progress] < 100 for supply_depot
             in self.get_units(Terran.SupplyDepot, alliance=PlayerRelative.SELF)]
        )

    def get_barracks_in_progress(self) -> int:
        return sum(
            [supply_depot[FeatureUnit.build_progress] < 100 for supply_depot
             in self.get_units(Terran.Barracks, alliance=PlayerRelative.SELF)]
        )

    def get_marines_in_progress(self) -> int:
        return sum(
            [b[FeatureUnit.order_length] > 0 for b in self.get_units(Terran.Barracks, alliance=PlayerRelative.SELF)]
        )

    def get_supply_depot_locations(self) -> np.ndarray:
        cc = self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF)[0]
        side_multiplier = (1 if self.player_on_left else -1)
        start_position = (cc.x + 4 * side_multiplier, cc.y)
        positions = []
        for x in range(0, 21, 2):
            for y in range(-5, 6, 5):
                positions.append([start_position[0] + x * side_multiplier, start_position[1] + y])
        positions = sorted(positions, key=lambda p: (p[0] * side_multiplier, abs(p[1])))
        return np.array(positions)

    def get_barracks_locations(self) -> np.ndarray:
        cc = self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF)[0]
        side_multiplier = (1 if self.player_on_left else -1)
        start_position = (cc.x - 1 * side_multiplier, cc.y + 2)
        positions = []
        for x in range(0, 26, 5):
            for y in range(-10, 6, 5):
                positions.append([start_position[0] + x * side_multiplier, start_position[1] + y])
        positions = sorted(positions, key=lambda p: (p[0] * side_multiplier, abs(p[1])))
        return np.array(positions)

    def get_enemy_base_location(self) -> np.ndarray:
        return np.array(self.base_locations[-1] if self.player_on_left else self.base_locations[0])

    def get_military_units(self) -> List:
        return self.get_units({Terran.Marine}, alliance=PlayerRelative.SELF)

    def has_any_military_units(self) -> ActionRequirement:
        return ActionRequirement(invalid=len(self.get_military_units()) == 0)

    def attack_enemy_base_location(self):
        units = self.get_military_units()
        if len(units) == 0:
            return None
        tags = [u.tag for u in units]
        self.logger.debug(f"Attack enemy base location: {self.enemy_base_location}")
        self.logger.debug(f"Attack enemy base location: {self.enemy_base_location}")
        return actions.RAW_FUNCTIONS.Attack_pt("now", tags, self.enemy_base_location)

    def attack_on_path(self):
        units = self.get_military_units()
        if len(units) == 0:
            return []

        path_points = self.attack_locations[1 - int(self.player_on_left)]
        unit_positions = np.array([[u.x, u.y] for u in units])
        target_positions = np.array([[t[0], t[1]] for t in path_points])

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
            attack_target_index = t_dx
            if t_dx + 1 < len(path_points):
                attack_target_index += 1
            else:
                self.reached_enemy_main = True
            target_location = np.array(path_points[attack_target_index])
            tags = [u.tag for u_idx, u in enumerate(units) if u_idx in unit_indices]
            attack_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", tags, target_location))
            self.logger.debug(f"Attack nearest target: {target_location}, tags: {tags}")
        return attack_list

    def attack_nearest_target(self) -> List:
        targets = self.get_units(alliance=PlayerRelative.ENEMY)
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
        if not self.reached_enemy_main:
            return self.attack_on_path()
        return self.attack_nearest_target() or [self.attack_enemy_base_location()]

    def stop_army(self) -> List:
        units = self.get_military_units()
        if len(units) == 0:
            return []
        tags = [u.tag for u in units]
        return [actions.RAW_FUNCTIONS.Stop_quick("now", tags)]

    def get_retreat_position(self) -> np.ndarray:
        if self.player_on_left:
            return np.array(self.base_locations[1]) + np.array([-10., 12.])
        else:
            return np.array(self.base_locations[2]) + np.array([10., -12.])

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
        supply_depots = self.get_units(Terran.SupplyDepot, alliance=PlayerRelative.SELF)
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

        self.action_space: Discrete
        mask = [True] * self.action_space.n
        for action_index, action_func in self.valid_action_mapping.items():
            action_requirement = action_func()
            mask[action_index] = action_requirement.can_do_instantly if self.pending_actions \
                else not action_requirement.invalid
        return np.array(mask)

    def get_score_cumulative(self):
        return self.raw_obs.observation.score_cumulative

    def get_score_by_category(self):
        return self.raw_obs.observation.score_by_category

    def should_surrender(self) -> bool:
        return len(self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF)) == 0

    def get_normalized_time(self) -> float:
        return self.get_game_step() / self.max_game_step

    def get_game_step(self) -> int:
        return self.raw_obs.observation.game_loop[0]

    def is_2nd_cc_built(self) -> bool:
        ccs = sorted(self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF), key=lambda c: c[FeatureUnit.x])
        return len(ccs) == 2 and ccs[1][FeatureUnit.build_progress] == 100

    def can_build_cc(self) -> ActionRequirement:
        action_requirement = self.get_requirements(Terran.CommandCenter)
        if self.cc_started:
            action_requirement.invalid = True
        return action_requirement

    def build_cc(self):
        location = np.array(self.base_locations[1 if self.player_on_left else 2])
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None
        self.cc_started = True
        return actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", worker[FeatureUnit.tag], location)

    def cancel_cc_rally(self):
        if self.cc_rally_canceled:
            return None
        cc = self.get_units(Terran.CommandCenter, alliance=PlayerRelative.SELF)[0]
        location = (cc[FeatureUnit.x], cc[FeatureUnit.y])
        self.cc_rally_canceled = True
        return actions.RAW_FUNCTIONS.Rally_Workers_pt("now", cc[FeatureUnit.tag], location)

    def can_do_nothing(self) -> ActionRequirement:
        return ActionRequirement()

    def do_nothing(self):
        return None

    def get_building_type_states(self) -> Dict[Terran, BuildingTypeState]:
        states: Dict[Terran, BuildingTypeState] = {}
        for building_type in self.building_types:
            alternate_building = {Terran.SupplyDepotLowered} if building_type == Terran.SupplyDepot else set()
            buildings = self.get_units({building_type} | alternate_building, alliance=PlayerRelative.SELF)
            if len(buildings) == 0:
                states[building_type] = BuildingTypeState.NOT_PRESENT
            elif any(building[FeatureUnit.build_progress] == 100 for building in buildings):
                states[building_type] = BuildingTypeState.IS_BUILT
            else:
                states[building_type] = BuildingTypeState.IS_BUILDING
        return states

    def should_do_finishing_attack(self):
        self.finishing_move_triggered |= self.get_normalized_time() >= self.time_to_finishing_move or \
                                         self.get_supply_taken() >= self.supply_to_finishing_move
        return self.finishing_move_triggered

    def get_action_requirements(self) -> Dict[ActionIndex, ActionRequirement]:
        return {action_index: able_func() for action_index, able_func in self.valid_action_mapping.items()}

    def register_episode_reward(self, reward: float):
        self.episode_rewards.append(reward)
        self.logger.info(f"Episode: {len(self.episode_rewards)},\t"
                         f"Length: {self.rl_step},\t"
                         f"Reward: {reward}")
        for n_mean in [5, 25, 100]:
            if len(self.episode_rewards) >= n_mean:
                self.logger.info(f"Mean last {n_mean} rewards: {np.mean(self.episode_rewards[-n_mean:])}")

        if self.output_path:
            try:
                with open(os.path.join(self.output_path, "episode_rewards.json"), "w") as f:
                    json.dump(self.episode_rewards, f)
            except Exception as e:
                self.logger.error(f"Error during saving episode rewards: '{e}'")

        self.update_difficulty_scheduler(reward)

    def update_difficulty_scheduler(self, score: float):
        if self.difficulty_scheduler is None:
            return None
        next_difficulty = self.difficulty_scheduler.report_score(score)
        self.settings["players"][1] = sc2_env.Bot(self.settings["players"][1].race, next_difficulty)

    def get_shaped_rewards(self) -> float:
        return functools.reduce(float.__add__, [
            reward_shaper.get_shaped_reward() for reward_shaper in self.reward_shapers
        ], 0.0)

    def get_production_building_count(self) -> int:
        buildings = self.get_units(self.production_building_types, alliance=PlayerRelative.SELF)
        return sum(building[FeatureUnit.build_progress] == 100 for building in buildings)

    def get_requirements(self, item: Union[Terran, Upgrades]) -> ActionRequirement:
        player = self.raw_obs.observation.player
        base_cost = get_item_cost(item)
        total_cost = base_cost.clone()
        for pending_action in self.pending_actions:
            if pending_action in action_to_unit and action_to_unit[pending_action] == item:
                break
            if pending_action in action_to_upgrade and item in action_to_upgrade[pending_action]:
                break
            total_cost += self.get_action_cost(pending_action)
        action_requirement = ActionRequirement()

        if player[Player.food_used] + total_cost.supply > supply_limit:
            action_requirement.invalid = True
        elif player[Player.food_used] + total_cost.supply > player[Player.food_cap]:
            action_requirement.buildings.append(Terran.SupplyDepot)
        if player[Player.minerals] < total_cost.minerals:
            action_requirement.minerals = True
        if player[Player.vespene] < total_cost.vespene and base_cost.vespene > 0:
            action_requirement.vespene = True

        building_requirement = get_building_requirement(item)
        if building_requirement.production:
            production_building_types = set(building_requirement.production)
            if len(self.get_units(production_building_types, alliance=PlayerRelative.SELF)) == 0:
                action_requirement.buildings.append(building_requirement.production[0])
            elif self.get_free_building(production_building_types) is None:
                action_requirement.queue = True

        for building_type in building_requirement.exists:
            building_state = self.building_states[Terran(building_type)]
            if building_state == BuildingTypeState.NOT_PRESENT:
                action_requirement.buildings.append(building_type)
            elif building_state == BuildingTypeState.IS_BUILDING:
                action_requirement.queue = True
        return action_requirement

    def update_enemy_race(self):
        if self.enemy_race != Race.random:
            return

        for unit in self.raw_obs.observation.raw_units:
            if unit.alliance == PlayerRelative.ENEMY:
                for race in (Terran, Zerg, Protoss):
                    try:
                        race(unit.unit_type)
                        if race == Terran:
                            self.enemy_race = Race.terran
                        elif race == Zerg:
                            self.enemy_race = Race.zerg
                        elif race == Protoss:
                            self.enemy_race = Race.protoss
                        else:
                            self.logger.warning(f"Unexpected enemy unit type: {unit.unit_type}")
                        break
                    except ValueError:
                        pass  # wrong race
                else:
                    self.logger.warning(f"Unexpected enemy unit type: {unit.unit_type}")
                if self.enemy_race != Race.random:
                    break

    def get_enemy_proximity(self) -> float:
        targets = self.get_units(alliance=PlayerRelative.ENEMY)
        targets = list(filter(lambda t: t.unit_type not in self.target_tags_to_ignore, targets))
        if len(targets) == 0:
            return 0.0
        target_positions = np.array([[t.x, t.y] for t in targets])
        if self.player_on_left:
            furthest_position = np.min(target_positions[:, 1])
            relative_position = (self.base_locations[-1][1] - furthest_position) / \
                                (self.base_locations[-1][1] - self.base_locations[0][1])
            return relative_position
        else:
            furthest_position = np.max(target_positions[:, 1])
            relative_position = (furthest_position - self.base_locations[0][1]) / \
                                (self.base_locations[-1][1] - self.base_locations[0][1])
            return relative_position

    def can_build_refinery(self) -> ActionRequirement:
        action_requirement = self.get_requirements(Terran.Refinery)
        if self.refinery_index >= self.max_refineries:
            action_requirement.invalid = True
        return action_requirement

    def build_refinery(self):
        if self.refinery_index >= len(self.geysers):
            self.logger.warning(f"Geyser not found")
            return None
        geyser = self.geysers[self.refinery_index]
        location = np.array([geyser[FeatureUnit.x], geyser[FeatureUnit.y]])

        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None

        self.refinery_index += 1
        return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", worker[FeatureUnit.tag], geyser[FeatureUnit.tag])

    def get_geysers(self):
        all_geysers = self.get_units(Neutral.VespeneGeyser)
        base_locations_indices = [0, 1] if self.player_on_left else [3, 2]
        geysers = []
        for base_location_index in base_locations_indices:
            base_location = self.base_locations[base_location_index]
            for geyser in all_geysers:
                if (geyser.x - base_location[0]) ** 2 + (geyser.y - base_location[1]) ** 2 < 100:
                    geysers.append(geyser)
        return geysers

    def get_upgrade_building_locations(self) -> List[Tuple[int, int]]:
        return self.upgrade_building_locations[0 if self.player_on_left else 1]

    def can_build_engineering_bay(self) -> ActionRequirement:
        action_requirement = self.get_requirements(Terran.EngineeringBay)
        locations = self.get_upgrade_building_locations()
        if self.upgrade_building_index >= len(locations):
            action_requirement.invalid = True
        if self.engineering_bay_index >= building_limits[Terran.EngineeringBay]:
            action_requirement.invalid = True
        return action_requirement

    def build_engineering_bay(self):
        locations = self.get_upgrade_building_locations()
        if self.upgrade_building_index >= len(locations):
            self.logger.warning("No upgrade building location left")
            return None
        if self.engineering_bay_index >= building_limits[Terran.EngineeringBay]:
            self.logger.warning("Max engineering bay count reached")
            return None

        location = locations[self.upgrade_building_index]

        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None

        self.upgrade_building_index += 1
        self.engineering_bay_index += 1
        return actions.RAW_FUNCTIONS.Build_EngineeringBay_pt("now", worker[FeatureUnit.tag], location)

    def can_research_infantry_weapons(self) -> ActionRequirement:
        completed_upgrades = self.raw_obs.observation.upgrades
        if Upgrades.TerranInfantryWeaponsLevel3 in completed_upgrades:
            return ActionRequirement(invalid=True)

        bays = self.get_units(Terran.EngineeringBay)
        for bay in bays:
            if bay.order_id_0 in {457, 458, 459, 456}:
                return ActionRequirement(invalid=True)

        if Upgrades.TerranInfantryWeaponsLevel2 in completed_upgrades:
            return self.get_requirements(Upgrades.TerranInfantryWeaponsLevel3)
        if Upgrades.TerranInfantryWeaponsLevel1 in completed_upgrades:
            return self.get_requirements(Upgrades.TerranInfantryWeaponsLevel2)
        return self.get_requirements(Upgrades.TerranInfantryWeaponsLevel1)

    def research_infantry_weapons(self):
        bay = self.get_free_building(Terran.EngineeringBay)
        if bay is None:
            self.logger.warning(f"Free Engineering Bay not found")
            return None

        return actions.RAW_FUNCTIONS.Research_TerranInfantryWeapons_quick("now", bay.tag)

    def can_research_infantry_armor(self) -> ActionRequirement:
        completed_upgrades = self.raw_obs.observation.upgrades
        if Upgrades.TerranInfantryArmorsLevel3 in completed_upgrades:
            return ActionRequirement(invalid=True)

        bays = self.get_units(Terran.EngineeringBay)
        for bay in bays:
            if bay.order_id_0 in {453, 454, 455, 452}:
                return ActionRequirement(invalid=True)

        if Upgrades.TerranInfantryArmorsLevel2 in completed_upgrades:
            return self.get_requirements(Upgrades.TerranInfantryArmorsLevel3)
        if Upgrades.TerranInfantryArmorsLevel1 in completed_upgrades:
            return self.get_requirements(Upgrades.TerranInfantryArmorsLevel2)
        return self.get_requirements(Upgrades.TerranInfantryArmorsLevel1)

    def research_infantry_armor(self):
        bay = self.get_free_building(Terran.EngineeringBay)
        if bay is None:
            self.logger.warning(f"Free Engineering Bay not found")
            return None

        return actions.RAW_FUNCTIONS.Research_TerranInfantryArmor_quick("now", bay.tag)

    def can_build_factory(self):
        action_requirement = self.get_requirements(Terran.Factory)
        if self.production_building_index >= len(self.production_buildings_locations):
            action_requirement.invalid = True
        if len(self.get_units(Terran.Factory)) >= building_limits[Terran.Factory]:
            action_requirement.invalid = True
        return action_requirement

    def build_factory(self):
        location = self.production_buildings_locations[self.production_building_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return []

        self.production_building_index += 1
        return actions.RAW_FUNCTIONS.Build_Factory_pt("now", worker[FeatureUnit.tag], location)

    def can_build_armory(self) -> ActionRequirement:
        action_requirement = self.get_requirements(Terran.Armory)
        locations = self.get_upgrade_building_locations()
        if self.upgrade_building_index >= len(locations):
            action_requirement.invalid = True
        if self.armory_index >= building_limits[Terran.Armory]:
            action_requirement.invalid = True
        return action_requirement

    def build_armory(self):
        locations = self.get_upgrade_building_locations()
        if self.upgrade_building_index >= len(locations):
            self.logger.warning("No upgrade building location left")
            return None
        if self.armory_index >= building_limits[Terran.Armory]:
            self.logger.warning("Max engineering bay count reached")
            return None

        location = locations[self.upgrade_building_index]

        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning(f"Free worker not found")
            return None

        self.upgrade_building_index += 1
        self.armory_index += 1
        return actions.RAW_FUNCTIONS.Build_Armory_pt("now", worker[FeatureUnit.tag], location)

    def get_action_cost(self, action_index: ActionIndex) -> Cost:
        cost = self.get_incremental_upgrade_cost(action_index)
        if cost is not None:
            return cost

        item_type = action_to_unit.get(action_index, -1)
        if item_type == -1:
            item_type = action_to_upgrade.get(action_index, -1)
        return unit_to_cost.get(item_type, Cost())

    def get_incremental_upgrade_cost(self, action_index: ActionIndex) -> Optional[Cost]:
        completed_upgrades = self.raw_obs.observation.upgrades
        if action_index == ActionIndex.RESEARCH_INFANTRY_WEAPONS:
            if Upgrades.TerranInfantryWeaponsLevel2 in completed_upgrades:
                return upgrade_to_cost[Upgrades.TerranInfantryWeaponsLevel3]
            if Upgrades.TerranInfantryWeaponsLevel1 in completed_upgrades:
                return upgrade_to_cost[Upgrades.TerranInfantryWeaponsLevel2]
            return upgrade_to_cost[Upgrades.TerranInfantryWeaponsLevel1]

        if action_index == ActionIndex.RESEARCH_INFANTRY_ARMOR:
            if Upgrades.TerranInfantryArmorsLevel2 in completed_upgrades:
                return upgrade_to_cost[Upgrades.TerranInfantryArmorsLevel3]
            if Upgrades.TerranInfantryArmorsLevel1 in completed_upgrades:
                return upgrade_to_cost[Upgrades.TerranInfantryArmorsLevel2]
            return upgrade_to_cost[Upgrades.TerranInfantryArmorsLevel1]
