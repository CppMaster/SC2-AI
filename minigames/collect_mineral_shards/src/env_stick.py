from typing import List

import numpy as np
from pysc2.lib import actions
from minigames.collect_mineral_shards.src.env import CollectMineralShardsEnv


class CollectMineralShardsStickEnv(CollectMineralShardsEnv):

    def get_actions(self, action: np.ndarray) -> List:
        np_action = action.reshape((2, 2))
        minerals = self.get_minerals(self.last_raw_obs)
        minerals_positions = np.array([[mineral.x, mineral.y] for mineral in minerals])
        mapped_actions = []
        for idx, tag in enumerate(self.unit_tags):
            pos = np_action[idx] * self.resolution + self.resolution * 0.5
            target_pos = minerals_positions[np.sum(np.power(minerals_positions - pos, 2), axis=1).argmin()]
            mapped_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", tag, target_pos))
        return mapped_actions

    @staticmethod
    def get_minerals(raw_obs):
        return [unit for unit in raw_obs.observation.raw_units
                if unit.unit_type == CollectMineralShardsStickEnv.mineral_shard_tag]
