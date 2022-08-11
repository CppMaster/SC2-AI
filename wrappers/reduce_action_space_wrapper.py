from typing import List

import gym
import numpy as np
from gym.spaces import Discrete


class ReduceActionSpaceWrapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env, masked_env: gym.Env, valid_actions: List[int]):
        super().__init__(env)
        assert isinstance(env.action_space, Discrete), "Only discrete action spaces are valid"
        self.masked_env = masked_env
        self.valid_actions = valid_actions
        self.inverse_valid_actions = {a: i for i, a in enumerate(valid_actions)}
        self.action_space = Discrete(len(self.valid_actions))

    def action(self, action):
        return self.valid_actions[action]

    def reverse_action(self, action):
        raise NotImplementedError

    def action_masks(self) -> np.ndarray:

        original_action_mask = self.masked_env.action_masks()
        if self.is_discrete:

            self.action_space: Discrete
            mask = [True] * self.action_space.n
            for action_index, original_action_index in enumerate(self.valid_actions):
                mask[action_index] = original_action_mask[original_action_index]
            return np.array(mask)
        else:
            raise NotImplementedError("Action mask not implemented for non-discrete action space")