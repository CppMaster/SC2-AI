import random
import sys
from absl import flags
import numpy as np

from stable_baselines3.common.monitor import Monitor


from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv, ActionIndex

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = PlannedActionEnv(step_mul=1, realtime=False)
monitor_env = Monitor(env)

while True:
    done = False
    obs = monitor_env.reset()
    while not done:
        action_mask = env.action_masks()
        valid_indices = [idx for idx, mask in enumerate(action_mask) if mask]
        action = random.choice(valid_indices)
        print(f"Chosen action: {ActionIndex(action).name}")
        obs, rewards, done, info = monitor_env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
