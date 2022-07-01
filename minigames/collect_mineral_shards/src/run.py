from stable_baselines3 import PPO

from minigames.collect_mineral_shards.src.env import CollectMineralShardsEnv

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


model = PPO.load("minigames/collect_mineral_shards/eval/best_model.zip")
env = CollectMineralShardsEnv(step_mul=8, realtime=True)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
