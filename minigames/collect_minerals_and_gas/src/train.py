import sys
from absl import flags
import logging

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from torch import nn

from callbacks.stop_training_on_no_model_training_improvement import StopTrainingOnNoModelTrainingImprovement
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

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

suffix = "found-values_step-mul-1"

found_reward_values = {
    'reward_scale': 0.00443894294414247,
    'worker_active_reward_scale': 0.5233561577446618,
    'supply_taken_reward_scale': 1.9013889305546567,
    'supply_depot_reward_scale': 1.1966878841334392,
    'supply_free_margin': 7,
    'cc_reward_scale': 0.4619790372759725,
    'cc_time_margin': 267.80016059275977,
    'refinery_reward_scale': 1.2306796817224745,
    'refinery_worker_slots_margin': 5,
    'refinery_suboptimal_worker_slot_weight': 0.2560175654282947
}

env = CollectMineralAndGasDiscreteEnv(step_mul=1, realtime=False)
# env = CollectMineralAndGasEnv(step_mul=8, realtime=False)
env = Monitor(env)
env = WorkersActiveRewardWrapper(env,
                                 mineral_reward=100. * found_reward_values["worker_active_reward_scale"],
                                 lesser_mineral_reward=50. * found_reward_values["worker_active_reward_scale"],
                                 gas_reward=75. * found_reward_values["worker_active_reward_scale"])
env = SupplyTakenRewardWrapper(env, reward_diff=100. * found_reward_values["supply_taken_reward_scale"])
env = SupplyDepotRewardWrapper(env,
                               reward_diff=100. * found_reward_values["supply_depot_reward_scale"],
                               free_supply_margin=found_reward_values["supply_free_margin"])
env = CommandCenterRewardWrapper(env,
                                 reward_diff=10. * found_reward_values["cc_reward_scale"],
                                 time_margin=found_reward_values["cc_time_margin"])
env = RefineryRewardWrapper(env,
                            reward_diff=100. * found_reward_values["refinery_reward_scale"],
                            workers_slots_margin=found_reward_values["refinery_worker_slots_margin"],
                            suboptimal_worker_slot_weight=found_reward_values["refinery_suboptimal_worker_slot_weight"])
env = RewardScaleWrapper(env, found_reward_values["reward_scale"])

eval_path = f"minigames/collect_minerals_and_gas/results/eval/eval_logs_{suffix}"
# stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10, verbose=1)
# eval_callback = MaskableEvalCallback(
#     env, best_model_save_path=eval_path, log_path=eval_path, eval_freq=100000, deterministic=False, render=False,
#     callback_after_eval=stop_callback, n_eval_episodes=20
# )
callback = StopTrainingOnNoModelTrainingImprovement(max_no_improvement_evals=10, eval_every_n_step=10000, verbose=1,
                                                    min_evals=10)

model = MaskablePPO(
    "MlpPolicy", env, verbose=1,
    tensorboard_log=f"minigames/collect_minerals_and_gas/results/logs/logs_{suffix}",
    gamma=0.99, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=64, learning_rate=3e-4, normalize_advantage=True
)
# model = MaskablePPO.load("minigames/collect_minerals_and_gas/results/eval/eval_logs_reward-wrappers_cc-reward-wrapper_refinery-reward-wrapper/best_model.zip",
#                          env=env)
model.learn(10000000, callback=callback, reset_num_timesteps=True)
model.save(f"{eval_path}/last_model.zip")
