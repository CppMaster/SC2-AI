import sys
from absl import flags

from minigames.collect_mineral_shards.src.env import CollectMineralShardsEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = CollectMineralShardsEnv(step_mul=8, realtime=True)

obs = env.reset()
