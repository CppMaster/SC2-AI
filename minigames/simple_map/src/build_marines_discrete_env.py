from typing import List

import numpy as np
from gym.spaces import Discrete

from minigames.simple_map.src.build_marines_env import BuildMarinesEnv, ActionIndex


class BuildMarinesDiscreteEnv(BuildMarinesEnv):

    def __init__(self, step_mul: int = 8, realtime: bool = False):
        super().__init__(step_mul, realtime)
        self.action_space = Discrete(len(ActionIndex) + 1)

    def get_actions(self, action: int) -> List:
        mapped_actions = self.send_idle_workers_to_work()
        if action == ActionIndex.BUILD_MARINE:
            mapped_actions.append(self.build_marine())
        elif action == ActionIndex.BUILD_SCV:
            mapped_actions.append(self.build_scv())
        elif action == ActionIndex.BUILD_SUPPLY:
            mapped_actions.append(self.build_supply_depot())
        elif action == ActionIndex.BUILD_BARRACKS:
            mapped_actions.append(self.build_barracks())

        mapped_actions = list(filter(lambda x: x is not None, mapped_actions))
        return mapped_actions

    def action_masks(self) -> np.ndarray:
        mask = [True] * self.action_space.n
        mask[ActionIndex.BUILD_MARINE] = self.can_build_marine()
        mask[ActionIndex.BUILD_SCV] = self.can_build_scv()
        mask[ActionIndex.BUILD_SUPPLY] = self.can_build_supply_depot()
        mask[ActionIndex.BUILD_BARRACKS] = self.can_build_barracks()
        return np.array(mask)
