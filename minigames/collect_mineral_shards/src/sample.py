import sys
from absl import flags
import numpy as np
from minigames.collect_mineral_shards.src.env_stick import CollectMineralShardsStickEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = CollectMineralShardsStickEnv(step_mul=8, realtime=False)

episode_rewards = []
while True:
    done = False
    total_rewards = 0
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards
    episode_rewards.append(total_rewards)
    print(f"Episode reward: {total_rewards},\tAverage reward: {np.mean(episode_rewards)}")
