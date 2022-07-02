from typing import Union, Optional
import numpy as np
import gym
from stable_baselines3.common.type_aliases import GymStepReturn


class RepeatActionUntilReward(gym.Wrapper):

    def __init__(self, env: gym.Env, frame_punish: float = -1.0, max_skip: Optional[int] = None):
        super(RepeatActionUntilReward, self).__init__(env=env)
        self.frame_punish = frame_punish
        self.max_skip = max_skip

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        total_step_reward = 0.0
        keep_skipping = True
        observation = None
        done = False
        info = {}
        n_skip = 0
        while keep_skipping:
            observation, reward, done, info = self.env.step(action)
            total_step_reward += reward + self.frame_punish
            n_skip += 1
            if reward > 0 or done or (self.max_skip is not None and n_skip > self.max_skip):
                keep_skipping = False

        return observation, total_step_reward, done, info

