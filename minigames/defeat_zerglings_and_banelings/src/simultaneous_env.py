from collections import defaultdict
from typing import List, Dict

import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SimultaneousEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, raw_resolution: int = 64, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.banelings = []
        self.zerglings = []
        self.raw_resolution = raw_resolution
        # 10 marines, 4 moves, 9 enemies, 1 noop
        self.action_space = spaces.MultiDiscrete([14] * 9)
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(19, 3),
            dtype=np.float
        )
        self.settings = {
            'map_name': "DefeatZerglingsAndBanelings",
            'players': [sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=self.raw_resolution),
            'realtime': False
        }

    def reset(self):
        if self.env is None:
            self.init_env()

        self.marines = []
        self.banelings = []
        self.zerglings = []

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((19, 3), dtype=np.float)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)
        self.marines = []
        self.banelings = []
        self.zerglings = []

        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x / self.raw_resolution, m.y / self.raw_resolution, m[2] / 45])

        for i, b in enumerate(banelings):
            self.banelings.append(b)
            obs[i+9] = np.array([b.x / self.raw_resolution, b.y / self.raw_resolution, b[2] / 30])

        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i+13] = np.array([z.x / self.raw_resolution, z.y / self.raw_resolution, z[2] / 35])

        return obs

    def step(self, action: np.ndarray):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        # each step will set the dictionary to emtpy
        return obs, reward, raw_obs.last(), {}

    def take_action(self, action: np.ndarray):

        attack_indices: Dict[int, List[int]] = defaultdict(list)

        action_mapped: List = []

        for u_idx, a_idx in enumerate(action):

            if a_idx == 0:
                pass
            elif a_idx == 1:
                action_mapped.append(self.move_up(u_idx))
            elif a_idx == 2:
                action_mapped.append(self.move_down(u_idx))
            elif a_idx == 3:
                action_mapped.append(self.move_left(u_idx))
            elif a_idx == 4:
                action_mapped.append(self.move_right(u_idx))
            else:
                target_index = a_idx - 5
                attack_indices[target_index].append(u_idx)

        for target_index, unit_indices in attack_indices.items():
            action_mapped.append(self.attack(unit_indices, target_index))

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def move_up(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y-2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except IndexError:
            pass
        except Exception as e:
            logger.warning(f"Action move_up not successful: {e}")
        return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y+2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except IndexError:
            pass
        except Exception as e:
            logger.warning(f"Action move_down not successful: {e}")
        return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x-2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except IndexError:
            pass
        except Exception as e:
            logger.warning(f"Action move_left not successful: {e}")
        return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x+2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except IndexError:
            pass
        except Exception as e:
            logger.warning(f"Action move_right not successful: {e}")
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, unit_indices: List[int], target_index: int):
        try:
            selected_tags = [unit.tag for uidx, unit in enumerate(self.marines) if uidx in unit_indices]
            if len(selected_tags) == 0:
                return actions.RAW_FUNCTIONS.no_op()
            if target_index > 3:
                # attack zerglings
                target = self.zerglings[target_index-4]
            else:
                target = self.banelings[target_index]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected_tags, target.tag)
        except IndexError:
            pass
        except Exception as e:
            logger.warning(f"Action attack not successful: {e}")
        return actions.RAW_FUNCTIONS.no_op()

    def get_units_by_type(self, obs, unit_type, player_relative=0):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == player_relative]

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass