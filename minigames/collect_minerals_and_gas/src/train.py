import sys
from absl import flags
import logging
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from torch import nn

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from minigames.collect_minerals_and_gas.src.env_dicrete import CollectMineralAndGasDiscreteEnv
from policies.actor_critic_action_elimination_policy import ActorCriticActionEliminationPolicy

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.INFO)

suffix = "worker_allocation"

env = CollectMineralAndGasDiscreteEnv(step_mul=8, realtime=False)
# env = CollectMineralAndGasEnv(step_mul=4, realtime=False)
env = Monitor(env)

eval_path = f"minigames/collect_minerals_and_gas/eval_logs_{suffix}"
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=60, min_evals=100, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=None, log_path=eval_path,
                             eval_freq=10000, deterministic=False, render=False,
                             callback_after_eval=stop_callback)

policy = ActorCriticActionEliminationPolicy
model = PPO(policy, env, verbose=1, tensorboard_log=f"minigames/collect_minerals_and_gas/logs_{suffix}", gamma=0.9999,
            policy_kwargs=dict(eliminated_actions_func=env.get_invalid_actions, activation_fn=nn.LeakyReLU))
# model = PPO.load("minigames/collect_minerals_and_gas/eval_logs_discrete16/best_model.zip", env=env)
model.learn(10000000, callback=eval_callback, reset_num_timesteps=False)
