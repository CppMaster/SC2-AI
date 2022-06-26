from stable_baselines3 import PPO
import logging

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from minigames.defeat_zerglings_and_banelings.src.fyr91_env import DZBEnv

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


logging.basicConfig(encoding='utf-8', level=logging.INFO)

env = DZBEnv(step_mul=1)
env = Monitor(env)
eval_path = "minigames/defeat_zerglings_and_banelings/eval"
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=eval_path, log_path=eval_path,
                             eval_freq=100, deterministic=False, render=False,
                             callback_after_eval=stop_callback)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="minigames/defeat_zerglings_and_banelings/logs")
# model = PPO.load("minigames/move_to_beacon/eval/best_model.zip", env=env)
model.learn(10000000, callback=eval_callback)

