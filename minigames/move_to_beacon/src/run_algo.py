import numpy as np
from stable_baselines3 import PPO

from minigames.move_to_beacon.src.env import MoveToBeaconEnv

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


model = PPO.load("minigames/move_to_beacon/logs/PPO_8/model_731136.zip")
env = MoveToBeaconEnv(step_mul=8, realtime=False)

obs = env.reset()
done = False

while not done:
    action = np.array([obs[2], obs[3]])
    obs, rewards, dones, info = env.step(action)
