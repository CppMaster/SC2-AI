import logging
from enum import IntEnum
import random
from typing import Optional, List, Set, Dict, Any

import gym
import numpy as np
from gym.spaces import MultiDiscrete, Box
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import features, actions
from pysc2.lib.features import FeatureUnit, Player
from pysc2.lib.units import Terran, Neutral

from minigames.collect_minerals_and_gas.src.env import OrderId


class ActionIndex(IntEnum):
    """Enumeration for action indices in the action space."""
    BUILD_MARINE = 0
    BUILD_SCV = 1
    BUILD_SUPPLY = 2
    BUILD_BARRACKS = 3


class ObservationIndex(IntEnum):
    """Enumeration for observation indices in the observation space."""
    MINERALS = 0  # scale 500
    SUPPLY_TAKEN = 1  # scale 150
    SUPPLY_ALL = 2  # scale 150
    SUPPLY_FREE = 3  # scale 16
    SCV_COUNT = 4  # scale 50
    TIME_LEFT = 5
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
    """
    Custom Gym environment for the BuildMarines minigame in StarCraft II.
    Implements logic for building marines, SCVs, supply depots, and barracks.

    Attributes
    ----------
    metadata : dict
        Metadata for rendering modes.
    max_game_step : int
        Maximum number of game steps.
    supply_depot_locations : np.ndarray
        Predefined locations for supply depots.
    barracks_locations : np.ndarray
        Predefined locations for barracks.
    scv_limit : int
        Maximum number of SCVs allowed.
    rally_position : np.ndarray
        Default rally position for units.
    """
    metadata: Dict[str, List[str]] = {'render.modes': ['human']}
    max_game_step: int = 14400
    supply_depot_locations: np.ndarray = np.array([
        [29, 29], [29, 44], [29, 31], [29, 42], [27, 29], [27, 44], [27, 31], [27, 42],
        [25, 29], [25, 44], [23, 29], [23, 44], [20, 29], [20, 44],
        [20, 35], [20, 33], [20, 31], [20, 38], [20, 40], [20, 42],
        [29, 33], [27, 33]
    ])
    barracks_locations: np.ndarray = np.array([
        [35, 34], [35, 38], [39, 34], [39, 38], [43, 34], [43, 38], [47, 34], [47, 38],
        [35, 30], [35, 42], [39, 30], [39, 42], [43, 30], [43, 42], [47, 30], [47, 42],
        [35, 26], [35, 46], [39, 26], [39, 46], [43, 26], [43, 46], [47, 26], [47, 46],
    ])
    scv_limit: int = 28
    rally_position: np.ndarray = np.array([20, 36])

    def __init__(self, step_mul: int = 8, realtime: bool = False) -> None:
        """
        Initialize the BuildMarinesEnv environment.

        Parameters
        ----------
        step_mul : int, optional
            Number of game steps per agent step (default is 8).
        realtime : bool, optional
            Whether to run in real-time mode (default is False).
        """
        self.settings: Dict[str, Any] = {
            'map_name': "BuildMarines",
            'players': [sc2_env.Agent(sc2_env.Race.terran)],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                crop_to_playable_area=True
            ),
            'realtime': realtime,
            'step_mul': step_mul
        }
        self.action_space: MultiDiscrete = MultiDiscrete([2] * len(ActionIndex))
        self.observation_space: Box = Box(low=0.0, high=1.0, shape=(len(ObservationIndex),))
        self.env: Optional[SC2Env] = None
        self.logger: logging.Logger = logging.getLogger("BuildMarinesEnv")
        self.raw_obs: Any = None

        self.supply_depot_index: int = 0
        self.barracks_index: int = 0
        self.rallies_set: Set[int] = set()

    def init_env(self) -> None:
        """
        Initialize the SC2 environment.
        """
        self.env = sc2_env.SC2Env(**self.settings)

    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial observation.

        Returns
        -------
        np.ndarray
            The initial normalized observation vector.
        """
        if self.env is not None:
            self.env.close()
        self.init_env()

        self.supply_depot_index = 0
        self.barracks_index = 0
        self.rallies_set = set()

        self.raw_obs = self.env.reset()[0]
        return self.get_derived_obs()

    def step(self, action: Optional[np.ndarray] = None) -> tuple:
        """
        Take an action in the environment.

        Parameters
        ----------
        action : np.ndarray or None, optional
            The action to take (default is None).

        Returns
        -------
        tuple
            (observation, reward, done, info)
        """
        self.raw_obs = self.env.step(self.get_actions(action))[0]
        derived_obs = self.get_derived_obs()
        return derived_obs, self.raw_obs.reward, self.raw_obs.last(), {}

    def get_actions(self, action: np.ndarray) -> List:
        """
        Map the action array to SC2 actions.

        Parameters
        ----------
        action : np.ndarray
            The action array.

        Returns
        -------
        list
            List of SC2 actions.
        """
        mapped_actions: List = self.send_idle_workers_to_work()

        if action[ActionIndex.BUILD_MARINE]:
            marine_action = self.build_marine()
            if marine_action:
                mapped_actions.append(marine_action)
        if action[ActionIndex.BUILD_SCV]:
            scv_action = self.build_scv()
            if scv_action:
                mapped_actions.append(scv_action)
        if action[ActionIndex.BUILD_SUPPLY]:
            supply_action = self.build_supply_depot()
            if supply_action:
                mapped_actions.append(supply_action)
        if action[ActionIndex.BUILD_BARRACKS]:
            barracks_action = self.build_barracks()
            if barracks_action:
                mapped_actions.append(barracks_action)

        return [a for a in mapped_actions if a is not None]

    def send_idle_workers_to_work(self) -> List:
        """
        Send idle SCVs to gather minerals.

        Returns
        -------
        list
            List of SC2 actions for idle workers.
        """
        idle_scvs = [u for u in self.get_units(Terran.SCV) if u[FeatureUnit.order_length] == 0]
        mineral = self.get_random_mineral()
        if mineral is None:
            return []
        return [actions.RAW_FUNCTIONS.Harvest_Gather_unit(
            "now", idle_scv[FeatureUnit.tag], mineral[FeatureUnit.tag]) for idle_scv in idle_scvs]

    def can_build_scv(self) -> bool:
        """
        Check if an SCV can be built.

        Returns
        -------
        bool
            True if an SCV can be built, False otherwise.
        """
        player = self.raw_obs.observation.player
        if player[Player.food_cap] - player[Player.food_used] < 1:
            return False
        if player[Player.minerals] < 50:
            return False
        if len(self.get_units(Terran.SCV)) >= self.scv_limit:
            return False
        cc = self.get_units(Terran.CommandCenter)[0]
        if cc[FeatureUnit.order_length] > 0:
            return False
        return True

    def build_scv(self) -> Optional[Any]:
        """
        Return the action to build an SCV if possible.

        Returns
        -------
        object or None
            The SC2 action to build an SCV, or None if not possible.
        """
        if self.can_build_scv():
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now", self.get_units(Terran.CommandCenter)[0].tag)
        return None

    def can_build_supply_depot(self) -> bool:
        """
        Check if a supply depot can be built.

        Returns
        -------
        bool
            True if a supply depot can be built, False otherwise.
        """
        player = self.raw_obs.observation.player
        if player[Player.minerals] < 100:
            return False
        if self.supply_depot_index >= len(self.supply_depot_locations):
            return False
        return True

    def build_supply_depot(self) -> Optional[Any]:
        """
        Return the action to build a supply depot if possible.

        Returns
        -------
        object or None
            The SC2 action to build a supply depot, or None if not possible.
        """
        if not self.can_build_supply_depot():
            return None
        location = self.supply_depot_locations[self.supply_depot_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning("Free worker not found for supply depot.")
            return None
        self.supply_depot_index += 1
        return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", worker[FeatureUnit.tag], location)

    def can_build_barracks(self) -> bool:
        """
        Check if a barracks can be built.

        Returns
        -------
        bool
            True if a barracks can be built, False otherwise.
        """
        player = self.raw_obs.observation.player
        if player[Player.minerals] < 150:
            return False
        if self.barracks_index >= len(self.barracks_locations):
            return False
        return True

    def build_barracks(self) -> Optional[Any]:
        """
        Return the action to build a barracks if possible.

        Returns
        -------
        object or None
            The SC2 action to build a barracks, or None if not possible.
        """
        if not self.can_build_barracks():
            return None
        location = self.barracks_locations[self.barracks_index]
        worker = self.get_nearest_worker(location)
        if worker is None:
            self.logger.warning("Free worker not found for barracks.")
            return None
        self.barracks_index += 1
        return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", worker[FeatureUnit.tag], location)

    def get_free_barracks(self) -> Optional[Any]:
        """
        Return a free barracks that can train marines, or None if none are available.

        Returns
        -------
        object or None
            The free barracks unit, or None if none are available.
        """
        for b in self.get_units(Terran.Barracks):
            if b[FeatureUnit.build_progress] < 100:
                continue
            if b[FeatureUnit.order_length] > 0:
                continue
            return b
        return None

    def can_build_marine(self) -> bool:
        """
        Check if a marine can be built.

        Returns
        -------
        bool
            True if a marine can be built, False otherwise.
        """
        player = self.raw_obs.observation.player
        if player[Player.food_cap] - player[Player.food_used] < 1:
            return False
        if player[Player.minerals] < 50:
            return False
        if self.get_free_barracks() is None:
            return False
        return True

    def build_marine(self) -> Optional[Any]:
        """
        Return the action to build a marine if possible.

        Returns
        -------
        object or None
            The SC2 action to build a marine, or None if not possible.
        """
        if not self.can_build_marine():
            return None
        barracks = self.get_free_barracks()
        if barracks is None:
            self.logger.warning("Free barracks not found for marine.")
            return None
        return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

    def get_random_mineral(self) -> Optional[Any]:
        """
        Return a random mineral field unit, or None if none are available.

        Returns
        -------
        object or None
            A random mineral field unit, or None if none are available.
        """
        minerals = self.get_units(Neutral.MineralField)
        if not minerals:
            return None
        return random.choice(minerals)

    def get_nearest_worker(self, location: np.ndarray) -> Optional[Any]:
        """
        Return the nearest available worker to a location.

        Parameters
        ----------
        location : np.ndarray
            The target location.

        Returns
        -------
        object or None
            The nearest worker unit or None.
        """
        all_workers = self.get_units(Terran.SCV)
        idle_workers = [u for u in all_workers if u[FeatureUnit.order_length] == 0]
        worker = self.get_nearest_worker_from_list(location, idle_workers)
        if worker is not None:
            return worker
        workers_mining = [u for u in all_workers if u[FeatureUnit.order_id_0] == OrderId.HARVEST_MINERALS]
        worker = self.get_nearest_worker_from_list(location, workers_mining)
        if worker is not None:
            return worker
        return None

    @staticmethod
    def get_nearest_worker_from_list(location: np.ndarray, workers: List) -> Optional[Any]:
        """
        Return the nearest worker from a list to a location.

        Parameters
        ----------
        location : np.ndarray
            The target location.
        workers : list
            List of worker units.

        Returns
        -------
        object or None
            The nearest worker unit or None.
        """
        if not workers:
            return None
        worker_positions = np.array([[worker[FeatureUnit.x], worker[FeatureUnit.y]] for worker in workers])
        return workers[np.sum(np.power(worker_positions - location, 2), axis=1).argmin()]

    def get_units(self, unit_type: int) -> List[Any]:
        """
        Return all units of a given type.

        Parameters
        ----------
        unit_type : int
            The unit type.

        Returns
        -------
        list
            List of units of the given type.
        """
        return [unit for unit in self.raw_obs.observation.raw_units if unit.unit_type == unit_type]

    def get_derived_obs(self) -> np.ndarray:
        """
        Compute the derived observation vector for the agent.

        Returns
        -------
        np.ndarray
            The normalized observation vector.
        """
        obs = np.zeros(shape=self.observation_space.shape)
        player = self.raw_obs.observation.player
        obs[ObservationIndex.MINERALS] = player[Player.minerals] / 500
        obs[ObservationIndex.SUPPLY_TAKEN] = player[Player.food_used] / 150
        obs[ObservationIndex.SUPPLY_ALL] = player[Player.food_cap] / 150
        obs[ObservationIndex.SUPPLY_FREE] = (player[Player.food_cap] - player[Player.food_used]) / 16
        obs[ObservationIndex.SCV_COUNT] = len(self.get_units(Terran.SCV)) / self.scv_limit
        obs[ObservationIndex.TIME_LEFT] = (self.max_game_step - self.raw_obs.observation.game_loop) / self.max_game_step
        obs[ObservationIndex.SUPPLY_DEPOT_COUNT] = self.supply_depot_index / len(self.supply_depot_locations)
        obs[ObservationIndex.IS_SUPPLY_DEPOT_BUILDING] = self.get_supply_depots_in_progress()
        obs[ObservationIndex.BARRACKS_COUNT] = self.barracks_index / len(self.barracks_locations)
        obs[ObservationIndex.IS_BARRACKS_BUILDING] = self.get_barracks_in_progress()
        obs[ObservationIndex.CAN_BUILD_MARINE] = float(self.can_build_marine())
        obs[ObservationIndex.CAN_BUILD_SCV] = float(self.can_build_scv())
        obs[ObservationIndex.CAN_BUILD_BARRACKS] = float(self.can_build_barracks())
        obs[ObservationIndex.CAN_BUILD_SUPPLY_DEPOT] = float(self.can_build_supply_depot())
        obs[ObservationIndex.SCV_IN_PROGRESS] = float(self.get_svc_in_progress())
        obs[ObservationIndex.MARINES_IN_PROGRESS] = self.get_marines_in_progress() / len(self.barracks_locations)
        return obs

    def get_supply_taken(self) -> int:
        """
        Return the current supply taken.

        Returns
        -------
        int
            The current supply taken.
        """
        return self.raw_obs.observation.player[Player.food_used]

    def get_supply_cap(self) -> int:
        """
        Return the current supply cap.

        Returns
        -------
        int
            The current supply cap.
        """
        return self.raw_obs.observation.player[Player.food_cap]

    def get_expected_supply_cap(self) -> int:
        """
        Return the expected supply cap based on built supply depots.

        Returns
        -------
        int
            The expected supply cap.
        """
        return 15 + self.supply_depot_index * 8

    def get_supply_depots_in_progress(self) -> int:
        """
        Return the number of supply depots currently being built.

        Returns
        -------
        int
            The number of supply depots in progress.
        """
        return sum(
            supply_depot[FeatureUnit.build_progress] < 100 for supply_depot in self.get_units(Terran.SupplyDepot)
        )

    def get_barracks_in_progress(self) -> int:
        """
        Return the number of barracks currently being built.

        Returns
        -------
        int
            The number of barracks in progress.
        """
        return sum(
            barracks[FeatureUnit.build_progress] < 100 for barracks in self.get_units(Terran.Barracks)
        )

    def get_svc_in_progress(self) -> bool:
        """
        Return True if an SCV is currently being built.

        Returns
        -------
        bool
            True if an SCV is in progress, False otherwise.
        """
        return self.get_units(Terran.CommandCenter)[0][FeatureUnit.order_length] > 0

    def get_marines_in_progress(self) -> int:
        """
        Return the number of marines currently being trained.

        Returns
        -------
        int
            The number of marines in progress.
        """
        return sum(
            b[FeatureUnit.order_length] > 0 for b in self.get_units(Terran.Barracks)
        )

    def render(self, mode: str = "human") -> None:
        """
        Render the environment (not implemented).

        Parameters
        ----------
        mode : str, optional
            The mode to render with (default is "human").
        """
        pass

    def close(self) -> None:
        """
        Close the environment and free resources.
        """
        if self.env is not None:
            self.env.close()
        super().close()
