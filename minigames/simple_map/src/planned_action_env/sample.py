import logging
import random
import sys
from typing import Optional

from absl import flags
import numpy as np
from pysc2.env.sc2_env import Difficulty

from stable_baselines3.common.monitor import Monitor


from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv, ActionIndex

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

env = PlannedActionEnv(step_mul=4, realtime=False, difficulty=Difficulty.hard, time_to_finishing_move=0.8,
                       free_supply_margin_factor=1.0)
monitor_env = Monitor(env)


forced_action: Optional[ActionIndex] = ActionIndex.RESEARCH_INFANTRY_ARMOR


while True:
    done = False
    obs = monitor_env.reset()
    while not done:
        action_mask = env.action_masks()
        valid_indices = [idx for idx, mask in enumerate(action_mask) if mask]
        action = random.choice(valid_indices)
        obs, rewards, done, info = monitor_env.step(action)
