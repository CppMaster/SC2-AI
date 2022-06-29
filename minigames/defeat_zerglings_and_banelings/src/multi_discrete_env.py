import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MultiDiscreteEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "DefeatZerglingsAndBanelings",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64),
        'realtime': False
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.banelings = []
        self.zerglings = []
        # 10 marines, 4 moves, 9 enemies
        self.action_space = spaces.MultiDiscrete([10, 4 + 13])
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(19, 3),
            dtype=np.uint8
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.marines = []
        self.banelings = []
        self.zerglings = []

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((19, 3), dtype=np.uint8)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)
        self.marines = []
        self.banelings = []
        self.zerglings = []

        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, m[2]])

        for i, b in enumerate(banelings):
            self.banelings.append(b)
            obs[i+9] = np.array([b.x, b.y, b[2]])

        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i+13] = np.array([z.x, z.y, z[2]])

        return obs

    def step(self, action: np.ndarray):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        # each step will set the dictionary to emtpy
        return obs, reward, raw_obs.last(), {}

    def take_action(self, action):
        u_idx = action[0]
        a_idx = action[1]

        if a_idx == 0:
            action_mapped = self.move_up(u_idx)
        elif a_idx == 1:
            action_mapped = self.move_down(u_idx)
        elif a_idx == 2:
            action_mapped = self.move_left(u_idx)
        elif a_idx == 3:
            action_mapped = self.move_right(u_idx)
        else:
            action_mapped = self.attack(u_idx, a_idx - 4)

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

    def attack(self, aidx, eidx):
        try:
            selected = self.marines[aidx]
            if eidx > 3:
                # attack zerglings
                target = self.zerglings[eidx-4]
            else:
                target = self.banelings[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, target.tag)
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