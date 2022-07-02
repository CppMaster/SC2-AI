import sys
from absl import flags
import numpy as np
from stable_baselines3.common.monitor import Monitor

from minigames.colect_minerals_and_gas.src.env import CollectMineralAndGasEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = CollectMineralAndGasEnv(step_mul=1, realtime=False)
monitor_env = Monitor(env)
while True:
    done = False
    obs = env.reset()
    while not done:
        action = monitor_env.action_space.sample()
        obs, rewards, done, info = env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
