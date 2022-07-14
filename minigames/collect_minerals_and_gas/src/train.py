import sys
from absl import flags
import logging

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from torch import nn

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from minigames.collect_minerals_and_gas.src.env_dicrete import CollectMineralAndGasDiscreteEnv
from minigames.collect_minerals_and_gas.src.workers_active_reward_wrapper import WorkersActiveRewardWrapper
from wrappers.reward_scale_wrapper import RewardScaleWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.INFO)

suffix = "maskable2_lr-1e-5_bs-256_reward-wa_reward-scale-01"

env = CollectMineralAndGasDiscreteEnv(step_mul=8, realtime=False)
env = Monitor(env)
env = WorkersActiveRewardWrapper(env)
env = RewardScaleWrapper(env, 0.1)

eval_path = f"minigames/collect_minerals_and_gas/results/eval/eval_logs_{suffix}"
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=60, min_evals=100, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=eval_path, log_path=eval_path,
                             eval_freq=10000, deterministic=False, render=False,
                             callback_after_eval=stop_callback)

model = MaskablePPO("MlpPolicy", env, verbose=1,
                    tensorboard_log=f"minigames/collect_minerals_and_gas/results/logs/logs_{suffix}",
                    gamma=0.999, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True,),
                    batch_size=256, learning_rate=1e-5)
# model = PPO.load("minigames/collect_minerals_and_gas/eval_logs_discrete16/best_model.zip", env=env)
model.learn(10000000, callback=eval_callback, reset_num_timesteps=False)
