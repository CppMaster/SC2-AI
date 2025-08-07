from typing import List, Dict

import gym
import numpy as np
from gym.spaces import Discrete


class ReduceActionSpaceWrapper(gym.ActionWrapper):
    """
    Wrapper that reduces the action space to a subset of valid actions.
    
    This wrapper maps a reduced action space to the original action space,
    allowing for easier handling of action masking and valid action selection.

    Attributes
    ----------
    masked_env : gym.Env
        The environment that provides action masks.
    valid_actions : List[int]
        List of valid action indices in the original action space.
    inverse_valid_actions : Dict[int, int]
        Mapping from original action indices to reduced action indices.
    action_space : Discrete
        The reduced action space.
    """
    def __init__(self, env: gym.Env, masked_env: gym.Env, valid_actions: List[int]) -> None:
        """
        Initialize the ReduceActionSpaceWrapper.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        masked_env : gym.Env
            The environment that provides action masks.
        valid_actions : List[int]
            List of valid action indices in the original action space.
        """
        super().__init__(env)
        assert isinstance(env.action_space, Discrete), "Only discrete action spaces are valid"
        self.masked_env = masked_env
        self.valid_actions = valid_actions
        self.inverse_valid_actions = {a: i for i, a in enumerate(valid_actions)}
        self.action_space = Discrete(len(self.valid_actions))

    def action(self, action: int) -> int:
        """
        Map the reduced action to the original action space.

        Parameters
        ----------
        action : int
            Action in the reduced action space.

        Returns
        -------
        int
            Action in the original action space.
        """
        return self.valid_actions[action]

    def reverse_action(self, action: int) -> int:
        """
        Map the original action to the reduced action space.

        Parameters
        ----------
        action : int
            Action in the original action space.

        Returns
        -------
        int
            Action in the reduced action space.
        """
        raise NotImplementedError

    def action_masks(self) -> np.ndarray:
        """
        Get action masks for the reduced action space.

        Returns
        -------
        np.ndarray
            Boolean array indicating which actions are valid in the reduced space.
        """
        original_action_mask = self.masked_env.action_masks()
        
        # Handle discrete action spaces
        if isinstance(self.action_space, Discrete):
            mask = [True] * self.action_space.n
            for action_index, original_action_index in enumerate(self.valid_actions):
                mask[action_index] = original_action_mask[original_action_index]
            return np.array(mask)
        else:
            raise NotImplementedError("Action mask not implemented for non-discrete action space")