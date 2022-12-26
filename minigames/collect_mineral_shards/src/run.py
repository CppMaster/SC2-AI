import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from minigames.collect_mineral_shards.src.env import CollectMineralShardsEnv

import sys
from absl import flags

from minigames.collect_mineral_shards.src.env_stick import CollectMineralShardsStickEnv
from minigames.collect_mineral_shards.src.repeat_action_until_reward_wrapper import RepeatActionUntilReward

FLAGS = flags.FLAGS
FLAGS(sys.argv)


model = PPO.load("minigames/collect_mineral_shards/eval/best_model.zip")
env = CollectMineralShardsStickEnv(step_mul=1, realtime=False)
monitor_env = Monitor(env)
env = RepeatActionUntilReward(monitor_env, frame_punish=0.1)

while True:
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
