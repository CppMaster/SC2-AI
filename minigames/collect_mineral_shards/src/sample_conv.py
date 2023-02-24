import sys
from absl import flags
import numpy as np
from stable_baselines3.common.monitor import Monitor

from minigames.collect_mineral_shards.src.env_conv import CollectMineralShardsConvEnv
from minigames.collect_mineral_shards.src.image_to_image_policy import ImageToImagePolicy

FLAGS = flags.FLAGS
FLAGS(sys.argv)


grid_size = 32

env = CollectMineralShardsConvEnv(step_mul=2, realtime=False, resolution=grid_size)
monitor_env = Monitor(env)

policy = ImageToImagePolicy(n_conv_layers=7, kernel_size=5, value_features_layer=4,
                            grid_width=grid_size, grid_height=grid_size).to("cuda")

action = np.array([0, 1023])

while True:
    done = False
    obs = monitor_env.reset()
    while not done:
        observation = policy.obs_to_tensor(np.array([monitor_env.observation_space.sample()]))[0]
        action, value, log_prob = policy(observation)
        obs, rewards, done, info = monitor_env.step(action.cpu().numpy()[0])
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
