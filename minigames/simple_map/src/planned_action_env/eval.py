import logging
import sys
from absl import flags

import numpy as np
from pysc2.env.sc2_env import Difficulty
from sb3_contrib import MaskablePPO

from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv
from minigames.simple_map.src.planned_action_env.score_reward_wrapper import ScoreRewardShaper
from minigames.simple_map.src.planned_action_env.worker_reward_shaper import WorkerRewardShaper
from wrappers.stack_observations_actions_rewards_wrapper import StackObservationsActionRewardsWrapper, ValueStackConfig


FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

suffix = "more-observation_finishing-move-07"
output_path = f"minigames/simple_map/results/planned_action_logs/{suffix}"

env = PlannedActionEnv(step_mul=4, difficulty=Difficulty.hard, time_to_finishing_move=0.7,
                       reward_shapers=[ScoreRewardShaper(reward_diff=0.001, kill_factor=1.0, army_factor=1.0,
                                                         mined_factor=0.1, economy_factor=0.1),
                                       WorkerRewardShaper(reward_diff=0.001, optimal_reward=1.0, suboptimal_reward=0.1,
                                                          over_max_reward=-1.0)],
                       free_supply_margin_factor=1.0, output_path=output_path)
env = StackObservationsActionRewardsWrapper(env, reward_scale=1.0,
                                            observation_value_stack_config=ValueStackConfig(1, [10]),
                                            action_value_stack_config=ValueStackConfig(10, [50]),
                                            reward_value_stack_config=ValueStackConfig(10, [50]))


model = MaskablePPO.load(f"{output_path}/best_model_1.zip", env)

while True:
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs, action_masks=env.env.action_masks())
        obs, rewards, done, info = env.step(action)
    print(f"Episode reward: {env.env.episode_rewards[-1]},\tAverage reward: {np.mean(env.env.episode_rewards)}")
