import logging
import sys
from absl import flags
from sb3_contrib import MaskablePPO

from stable_baselines3.common.monitor import Monitor

from minigames.build_marines.src.build_marines_discrete_env import BuildMarinesDiscreteEnv
from torch import nn

from callbacks.stop_training_on_no_model_training_improvement import StopTrainingOnNoModelTrainingImprovement
from minigames.build_marines.src.supply_depot_reward_wrapper import SupplyDepotRewardWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)


logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

suffix = "discrete_mask_supply-depot-reward-1.0_step-mul-8_free-supply-margin-factor-1.5"
output_path = f"minigames/build_marines/results/logs/{suffix}"


env = BuildMarinesDiscreteEnv(step_mul=8, realtime=False)
env = Monitor(env)
env = SupplyDepotRewardWrapper(env, reward_diff=0.1, free_supply_margin_factor=1.5)

# callback = StopTrainingOnNoModelTrainingImprovement(max_no_improvement_evals=10, eval_every_n_step=10000, verbose=1,
#                                                     min_evals=10)

model = MaskablePPO(
    "MlpPolicy", env, verbose=1, tensorboard_log=output_path,
    gamma=0.99, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=64, learning_rate=3e-4, normalize_advantage=True
)
model.learn(10000000, callback=None, reset_num_timesteps=True)
model.save(f"{output_path}/last_model.zip")
