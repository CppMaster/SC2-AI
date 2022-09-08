import sys
import random

from absl import flags
import numpy as np
from pysc2.env.sc2_env import Difficulty
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from minigames.simple_map.src.instant_action_env.build_marines_discrete_env import BuildMarinesDiscreteEnv
from minigames.simple_map.src.instant_action_env.build_marines_env import BuildMarinesEnv, ActionIndex
from wrappers.add_action_and_reward_to_observation_wrapper import AddActionAndRewardToObservationWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)


# env = BuildMarinesEnv(step_mul=8, realtime=False)
env = BuildMarinesEnv(step_mul=4, realtime=False, is_discrete=True, difficulty=Difficulty.very_hard,
                      free_supply_margin_factor=1.5,
                      time_to_finishing_move=[0.5, 0.75], supply_to_finishing_move=[150, 200])
monitor_env = Monitor(env)
env = AddActionAndRewardToObservationWrapper(env)
exclude_actions = {ActionIndex.BUILD_CC}

# model = MaskablePPO.load("minigames/collect_minerals_and_gas/results/eval/eval_logs_reward-wrappers-workers-supply-taken-supply-depot/best_model.zip",
#                          env=env)

while True:
    done = False
    obs = monitor_env.reset()
    while not done:
        action_mask = env.action_masks()
        valid_indices = [idx for idx, mask in enumerate(action_mask) if mask and idx not in exclude_actions]
        action = random.choice(valid_indices)
        obs, rewards, done, info = monitor_env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
