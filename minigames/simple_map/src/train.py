import logging
import sys
from absl import flags
from sb3_contrib import MaskablePPO

from stable_baselines3.common.monitor import Monitor


from torch import nn

from callbacks.stop_training_on_no_model_training_improvement import StopTrainingOnNoModelTrainingImprovement
from minigames.simple_map.src.score_reward_wrapper import ScoreRewardWrapper
from minigames.simple_map.src.supply_depot_reward_wrapper import SupplyDepotRewardWrapper
from minigames.simple_map.src.build_marines_env import BuildMarinesEnv
from wrappers.reward_scale_wrapper import RewardScaleWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)


logging.basicConfig(encoding='utf-8', level=logging.INFO)

suffix = "supply-depot-reward-0.001_reward-scale-100._score-reward-0.01_gamma-0.9999"
output_path = f"minigames/simple_map/results/logs/{suffix}"


env = BuildMarinesEnv(step_mul=8, realtime=False, is_discrete=True)
env = Monitor(env)
env = RewardScaleWrapper(env, scale=100.)
env = SupplyDepotRewardWrapper(env, reward_diff=0.001, free_supply_margin_factor=1.5)
env = ScoreRewardWrapper(env, reward_diff=0.01)

# callback = StopTrainingOnNoModelTrainingImprovement(max_no_improvement_evals=10, eval_every_n_step=10000, verbose=1,
#                                                     min_evals=10)

model = MaskablePPO(
    "MlpPolicy", env, verbose=1, tensorboard_log=output_path,
    gamma=0.9999, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=64, learning_rate=3e-4, normalize_advantage=True
)
model.learn(10000000, callback=None, reset_num_timesteps=True)
model.save(f"{output_path}/last_model.zip")
