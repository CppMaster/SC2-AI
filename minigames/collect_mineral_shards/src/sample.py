import sys
from absl import flags

from minigames.collect_mineral_shards.src.env import CollectMineralShardsEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = CollectMineralShardsEnv(step_mul=8, realtime=False)

while True:
    done = False
    total_rewards = 0
    while not done:
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards
    print(f"Episode reward: {total_rewards}")
