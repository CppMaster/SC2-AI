import sys
from absl import flags
import numpy as np
from stable_baselines3.common.monitor import Monitor

from minigames.collect_mineral_shards.src.env_stick import CollectMineralShardsStickEnv
from minigames.collect_mineral_shards.src.repeat_action_until_reward_wrapper import RepeatActionUntilReward

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = CollectMineralShardsStickEnv(step_mul=1, realtime=False)
monitor_env = Monitor(env)
env = RepeatActionUntilReward(monitor_env)
while True:
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
