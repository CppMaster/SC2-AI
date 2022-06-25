from stable_baselines3 import PPO
import logging

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from minigames.move_to_beacon.src.env import MoveToBeaconEnv

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


logging.basicConfig(encoding='utf-8', level=logging.INFO)

env = MoveToBeaconEnv(step_mul=1)
eval_path = "minigames/move_to_beacon/eval"
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=eval_path, log_path=eval_path,
                             eval_freq=10000, deterministic=False, render=False,
                             callback_after_eval=stop_callback)

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="minigames/move_to_beacon/logs")
model = PPO.load("minigames/move_to_beacon/eval/best_model.zip", env=env)
model.learn(10000000, callback=eval_callback)

