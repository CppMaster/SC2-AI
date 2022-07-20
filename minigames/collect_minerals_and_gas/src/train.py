import sys
from absl import flags
import logging

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from torch import nn

from minigames.collect_minerals_and_gas.src.command_center_reward_wrapper import CommandCenterRewardWrapper
from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from minigames.collect_minerals_and_gas.src.env_dicrete import CollectMineralAndGasDiscreteEnv
from minigames.collect_minerals_and_gas.src.refinery_reward_wrapper import RefineryRewardWrapper
from minigames.collect_minerals_and_gas.src.supply_depot_reward_wrapper import SupplyDepotRewardWrapper
from minigames.collect_minerals_and_gas.src.supply_taken_reward_wrapper import SupplyTakenRewardWrapper
from minigames.collect_minerals_and_gas.src.workers_active_reward_wrapper import WorkersActiveRewardWrapper
from wrappers.reward_scale_wrapper import RewardScaleWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.INFO)

suffix = "all-reward-wrappers_step-mul-8"

env = CollectMineralAndGasDiscreteEnv(step_mul=8, realtime=False)
env = Monitor(env)
env = WorkersActiveRewardWrapper(env, mineral_reward=100., lesser_mineral_reward=50., gas_reward=75.)
env = SupplyTakenRewardWrapper(env, reward_diff=100.)
env = SupplyDepotRewardWrapper(env, reward_diff=100., free_supply_margin=6)
env = CommandCenterRewardWrapper(env, reward_diff=10.)
env = RefineryRewardWrapper(env, reward_diff=100., workers_slots_margin=4, suboptimal_worker_slot_weight=0.)
# env = RewardScaleWrapper(env, 0.1)

eval_path = f"minigames/collect_minerals_and_gas/results/eval/eval_logs_{suffix}"
stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=100, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=eval_path, log_path=eval_path,
                             eval_freq=10000, deterministic=False, render=False,
                             callback_after_eval=stop_callback, n_eval_episodes=10)

model = MaskablePPO(
    "MlpPolicy", env, verbose=1,
    tensorboard_log=f"minigames/collect_minerals_and_gas/results/logs/logs_{suffix}",
    gamma=0.99, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=64, learning_rate=3e-4, normalize_advantage=True
)
# model = MaskablePPO.load("minigames/collect_minerals_and_gas/results/eval/eval_logs_reward-wrappers_cc-reward-wrapper_refinery-reward-wrapper/best_model.zip",
#                          env=env)
model.learn(10000000, callback=eval_callback, reset_num_timesteps=True)
model.save(f"{eval_path}/last_model.zip")
