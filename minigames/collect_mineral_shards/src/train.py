from stable_baselines3 import PPO
import logging

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

import sys
from absl import flags

from minigames.collect_mineral_shards.src.env import CollectMineralShardsEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


logging.basicConfig(encoding='utf-8', level=logging.INFO)

env = CollectMineralShardsEnv(step_mul=8)
env = Monitor(env)
eval_path = "minigames/collect_mineral_shards/eval"
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=eval_path, log_path=eval_path,
                             eval_freq=10000, deterministic=False, render=False,
                             callback_after_eval=stop_callback)

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="minigames/collect_mineral_shards/logs",
            policy_kwargs=dict(net_arch=[128, dict(vf=[256], pi=[64])]))
# model = PPO.load("minigames/collect_mineral_shards/eval/best_model.zip", env=env)
model.learn(10000000, callback=eval_callback)
