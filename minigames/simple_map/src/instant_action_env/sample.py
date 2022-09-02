import sys
from absl import flags
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from minigames.simple_map.src.instant_action_env.build_marines_discrete_env import BuildMarinesDiscreteEnv
from minigames.simple_map.src.instant_action_env.build_marines_env import BuildMarinesEnv
from wrappers.add_action_and_reward_to_observation_wrapper import AddActionAndRewardToObservationWrapper

FLAGS = flags.FLAGS
FLAGS(sys.argv)


# env = BuildMarinesEnv(step_mul=8, realtime=False)
env = BuildMarinesEnv(step_mul=4, realtime=False, is_discrete=True, supple_depot_limit=3)
monitor_env = Monitor(env)
env = AddActionAndRewardToObservationWrapper(env)

# model = MaskablePPO.load("minigames/collect_minerals_and_gas/results/eval/eval_logs_reward-wrappers-workers-supply-taken-supply-depot/best_model.zip",
#                          env=env)

while True:
    done = False
    obs = monitor_env.reset()
    while not done:
        action = env.action_space.sample()
        # action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")