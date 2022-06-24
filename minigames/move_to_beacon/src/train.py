from stable_baselines3 import PPO
import logging

from stable_baselines3.common.callbacks import EvalCallback

from minigames.move_to_beacon.src.env import MoveToBeaconEnv

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


logging.basicConfig(encoding='utf-8', level=logging.INFO)

env = MoveToBeaconEnv(step_mul=4)
eval_env = MoveToBeaconEnv(step_mul=4)
eval_path = "minigames/move_to_beacon/eval"
eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path, log_path=eval_path,
                             eval_freq=100000, deterministic=False, render=False)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="minigames/move_to_beacon/logs")
# model = PPO.load("minigames/move_to_beacon/logs/PPO_8/model_731136.zip", env=env)
model.learn(10000000, callback=eval_callback)

