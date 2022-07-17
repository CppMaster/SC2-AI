import sys
from absl import flags
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from minigames.collect_minerals_and_gas.src.env import CollectMineralAndGasEnv
from minigames.collect_minerals_and_gas.src.env_dicrete import CollectMineralAndGasDiscreteEnv

FLAGS = flags.FLAGS
FLAGS(sys.argv)


env = CollectMineralAndGasDiscreteEnv(step_mul=8, realtime=False)
monitor_env = Monitor(env)

model = MaskablePPO.load("minigames/collect_minerals_and_gas/results/eval/eval_logs_reward-wrappers-workers-supply-taken-supply-depot/best_model.zip",
                         env=env)

while True:
    done = False
    obs = monitor_env.reset()
    while not done:
        # action = monitor_env.action_space.sample()
        action, _states = model.predict(obs)
        obs, rewards, done, info = monitor_env.step(action)
    print(f"Episode reward: {monitor_env.episode_returns[-1]},\tAverage reward: {np.mean(monitor_env.episode_returns)}")
