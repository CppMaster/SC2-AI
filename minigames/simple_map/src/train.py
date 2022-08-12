import logging
import sys
from absl import flags
from pysc2.env.sc2_env import Difficulty
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor

from torch import nn

from callbacks.log_episode_callabck import LogEpisodeCallback
from callbacks.stop_training_on_no_model_training_improvement import StopTrainingOnNoModelTrainingImprovement
from minigames.simple_map.src.score_reward_wrapper import ScoreRewardWrapper
from minigames.simple_map.src.supply_depot_reward_wrapper import SupplyDepotRewardWrapper
from minigames.simple_map.src.build_marines_env import BuildMarinesEnv, ActionIndex
from wrappers.add_action_and_reward_to_observation_wrapper import AddActionAndRewardToObservationWrapper
from wrappers.reduce_action_space_wrapper import ReduceActionSpaceWrapper
from wrappers.reward_scale_wrapper import RewardScaleWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.INFO)

suffix = "no-masking_difficulty-easy_recurrent"
output_path = f"minigames/simple_map/results/logs/{suffix}"

original_env = BuildMarinesEnv(step_mul=8, realtime=False, is_discrete=True, difficulty=Difficulty.easy)
env = Monitor(original_env)
env = RewardScaleWrapper(env, scale=100.)
env = SupplyDepotRewardWrapper(env, reward_diff=0.01, free_supply_margin_factor=1.5)
env = ScoreRewardWrapper(env, reward_diff=0.01, kill_factor=1.5)
env = ReduceActionSpaceWrapper(env, original_env,
                               [ActionIndex.BUILD_MARINE, ActionIndex.BUILD_SCV, ActionIndex.BUILD_SUPPLY,
                                ActionIndex.BUILD_BARRACKS, ActionIndex.ATTACK])
env = AddActionAndRewardToObservationWrapper(env, reward_scale=0.01)

# callback = StopTrainingOnNoModelTrainingImprovement(max_no_improvement_evals=10, eval_every_n_step=10000, verbose=1,
#                                                     min_evals=10)

model = RecurrentPPO(
    "MlpLstmPolicy", env, verbose=1, tensorboard_log=output_path,
    gamma=0.9999, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=64, learning_rate=1e-5, normalize_advantage=True
)
callback = LogEpisodeCallback(mean_episodes=[5, 25, 100])

model.learn(10000000, callback=callback, reset_num_timesteps=True)
model.save(f"{output_path}/last_model.zip")
