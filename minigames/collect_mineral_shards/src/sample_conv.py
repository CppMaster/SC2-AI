import sys
from absl import flags
import numpy as np
from stable_baselines3.common.monitor import Monitor

from minigames.collect_mineral_shards.src.env_conv import CollectMineralShardsConvEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = CollectMineralShardsConvEnv(step_mul=2, realtime=False)
monitor_env = Monitor(env)

action = np.array([0, 1023])

while True:
    done = False
    obs = monitor_env.reset()
    while not done:
        action = monitor_env.action_space.sample()
        obs, rewards, done, info = monitor_env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
