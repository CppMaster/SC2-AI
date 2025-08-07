import gym
import numpy as np
from gym.spaces import Discrete, Box
from typing import Tuple, Any, Dict


class AddActionAndRewardToObservationWrapper(gym.Wrapper):
    """
    Wrapper that adds action and reward information to the observation.
    
    This wrapper concatenates the original observation with a one-hot encoded
    action vector and the scaled reward value. This can be useful for providing
    additional context to the agent about its previous actions and rewards.

    Attributes
    ----------
    reward_scale : float
        Scaling factor for the reward value in the observation.
    """
    def __init__(self, env: gym.Env, reward_scale: float = 1.0) -> None:
        """
        Initialize the AddActionAndRewardToObservationWrapper.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        reward_scale : float, optional
            Scaling factor for the reward value (default is 1.0).
        """
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Only Box observation space handled"
        assert isinstance(env.action_space, Discrete), "Only Discrete action space handled"
        self.observation_space = Box(-1.0, 3.0, (env.observation_space.shape[0] + env.action_space.n + 1,))
        self.reward_scale = reward_scale

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment and return transformed observation.

        Parameters
        ----------
        action : int
            The action to take.

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict[str, Any]]
            (transformed_observation, reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)
        transformed_obs = np.concatenate([
            obs, 
            np.identity(self.env.action_space.n)[action],
            np.array([reward * self.reward_scale])
        ])
        return transformed_obs, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        """
        Reset the environment and return transformed observation.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the environment reset.

        Returns
        -------
        np.ndarray
            The transformed observation with zero action and reward.
        """
        obs = self.env.reset(**kwargs)
        transformed_obs = np.concatenate([
            obs, 
            np.zeros((self.env.action_space.n,)), 
            np.array([0])
        ])
        return transformed_obs
