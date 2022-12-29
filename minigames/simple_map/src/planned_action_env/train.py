import logging
import sys
from absl import flags
from pysc2.env.sc2_env import Difficulty, Race
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor

from torch import nn

from minigames.simple_map.src.planned_action_env.difficulty_scheduler import DifficultyScheduler
from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv
from minigames.simple_map.src.planned_action_env.score_reward_wrapper import ScoreRewardShaper
from minigames.simple_map.src.planned_action_env.supply_depot_reward_shaper import SupplyDepotRewardShaper
from minigames.simple_map.src.planned_action_env.worker_reward_shaper import WorkerRewardShaper
from wrappers.stack_observations_actions_rewards_wrapper import StackObservationsActionRewardsWrapper, ValueStackConfig

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

suffix = "infantry_upgrades_ebay_fix"
output_path = f"minigames/simple_map/results/planned_action_logs/{suffix}"

env = PlannedActionEnv(step_mul=4, difficulty=Difficulty.easy, enemy_race=Race.terran,
                       time_to_finishing_move=0.7,
                       reward_shapers=[ScoreRewardShaper(reward_diff=0.0001, kill_factor=1.0, army_factor=1.0,
                                                         mined_factor=0.1, economy_factor=0.1),
                                       WorkerRewardShaper(reward_diff=0.001, optimal_reward=1.0, suboptimal_reward=0.1,
                                                          over_max_reward=-1.0)],
                       free_supply_margin_factor=1.0, output_path=output_path,
                       difficulty_scheduler=DifficultyScheduler(min_mean_score=-.5, max_mean_score=0.5,
                                                                n_scores_mean=6),
                       max_refineries=1)
env = StackObservationsActionRewardsWrapper(env, reward_scale=1.0,
                                            observation_value_stack_config=ValueStackConfig(1, [10]),
                                            action_value_stack_config=ValueStackConfig(10, [50]),
                                            reward_value_stack_config=ValueStackConfig(10, [50]))

model = MaskablePPO(
    "MlpPolicy", env, verbose=1, tensorboard_log=output_path,
    gamma=0.9999, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=256, learning_rate=3e-4, normalize_advantage=True, n_steps=2 ** 14, n_epochs=10
)
# model = MaskablePPO.load(f"{output_path}/last_model.zip", env)

model.learn(100000000, reset_num_timesteps=True)
model.save(f"{output_path}/last_model.zip")
