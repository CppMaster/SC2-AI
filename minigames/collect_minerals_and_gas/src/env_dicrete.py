from typing import List

from gym.spaces import Discrete

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv, ActionIndex


class CollectMineralAndGasDiscreteEnv(CollectMineralAndGasEnv):

    def __init__(self, step_mul: int = 8, realtime: bool = False):
        super().__init__(step_mul, realtime)
        self.action_space = Discrete(len(ActionIndex) + 1)

    def get_actions(self, action: int) -> List:
        mapped_actions = self.send_idle_workers_to_minerals()
        if action == ActionIndex.BUILD_SCV_1:
            mapped_actions.append(self.build_scv(0))
        elif action == ActionIndex.BUILD_SCV_2:
            mapped_actions.append(self.build_scv(1))
        elif action == ActionIndex.BUILD_SUPPLY:
            mapped_actions.append(self.build_supply_depot())
        elif action == ActionIndex.BUILD_CC:
            mapped_actions.append(self.build_cc())
        elif action == ActionIndex.BUILD_REFINERY:
            mapped_actions.append(self.build_refinery())
        '''
        TODO:
        -Action for reassigning workers to opposite minerals or refinery
        '''
        mapped_actions = list(filter(lambda x: x is not None, mapped_actions))
        return mapped_actions
