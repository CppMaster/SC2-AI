import logging
import sys
from absl import flags
from pysc2.env.sc2_env import Difficulty
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor

from torch import nn


from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv
from minigames.simple_map.src.planned_action_env.score_reward_wrapper import ScoreRewardShaper

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

suffix = "init_diff-medium_score-reward-shaper_army-bonus-factor-10"
output_path = f"minigames/simple_map/results/planned_action_logs/{suffix}"

env = PlannedActionEnv(step_mul=2, difficulty=Difficulty.medium, time_to_finishing_move=0.6,
                       reward_shapers=[ScoreRewardShaper(reward_diff=0.001, kill_factor=1.0, army_bonus_factor=10.0)])

model = MaskablePPO(
    "MlpPolicy", env, verbose=1, tensorboard_log=output_path,
    gamma=0.9999, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=64, learning_rate=5e-6, normalize_advantage=True, n_steps=2400
)
# model = MaskablePPO.load("minigames/simple_map/results/logs/maskable_stack-obs_build-cc/medium-to-medium-hard_ep34.zip", env)

model.learn(10000000, reset_num_timesteps=True)
model.save(f"{output_path}/last_model.zip")
