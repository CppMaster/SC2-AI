from stable_baselines3 import PPO
import logging

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

import sys
from absl import flags

from minigames.collect_mineral_shards.src.double_action_ppo import DoubleActionPPO
from minigames.collect_mineral_shards.src.env_conv import CollectMineralShardsConvEnv
from minigames.collect_mineral_shards.src.image_to_image_policy import ImageToImagePolicy
from minigames.collect_mineral_shards.src.repeat_action_until_reward_wrapper import RepeatActionUntilReward

FLAGS = flags.FLAGS
FLAGS(sys.argv)

grid_size = 32

logging.basicConfig(encoding='utf-8', level=logging.INFO)

env = CollectMineralShardsConvEnv(step_mul=1, resolution=grid_size)
env = Monitor(env)

model = DoubleActionPPO(ImageToImagePolicy, env, verbose=1, tensorboard_log="minigames/collect_mineral_shards/logs",
                        gamma=0.99,
                        policy_kwargs=dict(n_conv_layers=7, kernel_size=5, value_features_layer=4,
                                           grid_width=grid_size, grid_height=grid_size))
model.learn(10000000, reset_num_timesteps=False, tb_log_name="DoubleActionPPO")
