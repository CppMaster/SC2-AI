import sys
from absl import flags
import logging
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from minigames.collect_minerals_and_gas.src.env_dicrete import CollectMineralAndGasDiscreteEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.INFO)

env = CollectMineralAndGasDiscreteEnv(step_mul=32, realtime=False)
# env = CollectMineralAndGasEnv(step_mul=4, realtime=False)
env = Monitor(env)

eval_path = "minigames/collect_minerals_and_gas/eval_logs_discrete16"
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=60, min_evals=100, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=eval_path, log_path=eval_path,
                             eval_freq=10000, deterministic=False, render=False,
                             callback_after_eval=stop_callback)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="minigames/collect_minerals_and_gas/logs_discrete16", gamma=0.99)
# model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="minigames/collect_minerals_and_gas/logs_discrete_dqn", gamma=0.99)
# model = PPO.load("minigames/collect_minerals_and_gas/eval/best_model.zip", env=env)
model.learn(10000000, callback=eval_callback, reset_num_timesteps=True)
