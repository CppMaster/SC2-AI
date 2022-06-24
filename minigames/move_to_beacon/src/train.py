from stable_baselines3 import PPO
import logging
from minigames.move_to_beacon.src.env import MoveToBeaconEnv

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


logging.basicConfig(encoding='utf-8', level=logging.INFO)

env = MoveToBeaconEnv(step_mul=32)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="minigames/move_to_beacon/logs")
model.learn(100000)

