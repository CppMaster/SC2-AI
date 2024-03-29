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
from minigames.simple_map.src.instant_action_env.attack_reward_wrapper import AttackRewardWrapper
from minigames.simple_map.src.instant_action_env.command_center_reward_wrapper import CommandCenterRewardWrapper
from minigames.simple_map.src.instant_action_env.score_reward_wrapper import ScoreRewardWrapper
from minigames.simple_map.src.instant_action_env.supply_depot_reward_wrapper import SupplyDepotRewardWrapper
from minigames.simple_map.src.instant_action_env.build_marines_env import BuildMarinesEnv, ActionIndex
from wrappers.add_action_and_reward_to_observation_wrapper import AddActionAndRewardToObservationWrapper
from wrappers.reduce_action_space_wrapper import ReduceActionSpaceWrapper
from wrappers.reward_scale_wrapper import RewardScaleWrapper
from wrappers.stack_observations_actions_rewards_wrapper import StackObservationsActionRewardsWrapper, ValueStackConfig

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logging.basicConfig(encoding='utf-8', level=logging.INFO)

suffix = "very-hard_finishing-move-05_score-reward-wrapper-1-2-mined-0_action-penalty-02_lr-0004_attack-time-offset-0"
output_path = f"minigames/simple_map/results/instant_action_logs/{suffix}"

original_env = BuildMarinesEnv(step_mul=4, realtime=False, is_discrete=True, difficulty=Difficulty.very_hard,
                               free_supply_margin_factor=1.5,
                               time_to_finishing_move=[0.50, 0.75], supply_to_finishing_move=[150, 200])
env = Monitor(original_env)
env = RewardScaleWrapper(env, scale=100.)
# env = SupplyDepotRewardWrapper(env, reward_diff=0.05, free_supply_margin_factor=1.5)
env.logger.setLevel(logging.DEBUG)
env = ScoreRewardWrapper(env, reward_diff=0.1, kill_factor=2.0, draw_plot=False, mined_factor=0.0)
# env = ReduceActionSpaceWrapper(env, original_env, [ActionIndex.BUILD_CC, ActionIndex.BUILD_SCV])
env = AttackRewardWrapper(env, reward_diff=0.1, time_offset=0.0, custom_multipliers={
    ActionIndex.ATTACK: 1.0, ActionIndex.RETREAT: -1.0, ActionIndex.STOP_ARMY: -0.2, ActionIndex.GATHER_ARMY: 0.5
}, action_penalty=0.2)
env.logger.setLevel(logging.DEBUG)
# env = CommandCenterRewardWrapper(env, reward_diff=5.0)
# env = AddActionAndRewardToObservationWrapper(env, reward_scale=0.01)
env = StackObservationsActionRewardsWrapper(
    env, reward_scale=0.01,
    observation_value_stack_config=ValueStackConfig(1, [5]),
    action_value_stack_config=ValueStackConfig(10, [20]),
    reward_value_stack_config=ValueStackConfig(20, [50])
)

# callback = StopTrainingOnNoModelTrainingImprovement(max_no_improvement_evals=10, eval_every_n_step=10000, verbose=1,
#                                                     min_evals=10)

model = MaskablePPO(
    "MlpPolicy", env, verbose=1, tensorboard_log=output_path,
    gamma=0.9999, policy_kwargs=dict(activation_fn=nn.LeakyReLU, ortho_init=True),
    batch_size=64, learning_rate=3e-4, normalize_advantage=True, n_steps=10000
)
# model = MaskablePPO.load("minigames/simple_map/results/logs/maskable_stack-obs_build-cc/medium-to-medium-hard_ep34.zip", env)
callback = LogEpisodeCallback(mean_episodes=[5, 25, 100])

model.learn(10000000, callback=callback, reset_num_timesteps=True)
model.save(f"{output_path}/last_model.zip")
