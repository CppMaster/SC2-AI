from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
import gym
import numpy as np
from gym.spaces import Discrete, Box

from utils.value.value_stack import ValueStack

@dataclass
class ValueStackConfig:
    """
    Configuration for value stacking in the wrapper.
    
    This dataclass defines how values should be stacked and averaged
    for observations, actions, and rewards.
    
    Attributes
    ----------
    n_last_values : int
        Number of last values to keep (default is 1).
    n_average_prev_last_values : Optional[List[int]]
        List of window sizes for averaging previous values (default is None).
    """
    n_last_values: int = 1
    n_average_prev_last_values: Optional[List[int]] = None

class StackObservationsActionRewardsWrapper(gym.Wrapper):
    """
    Wrapper that stacks observations, actions, and rewards over time.
    
    This wrapper maintains stacks of recent observations, actions, and rewards,
    allowing the agent to have access to historical information. This can be
    useful for environments where temporal context is important.

    Attributes
    ----------
    observation_value_stack : ValueStack
        Stack for storing recent observations.
    action_value_stack : ValueStack
        Stack for storing recent actions (one-hot encoded).
    reward_value_stack : ValueStack
        Stack for storing recent rewards.
    reward_scale : float
        Scaling factor for rewards.
    """
    def __init__(self, env: gym.Env, reward_scale: float = 1.0,
                 observation_value_stack_config: ValueStackConfig = ValueStackConfig(1, [5]),
                 action_value_stack_config: ValueStackConfig = ValueStackConfig(10, [20]),
                 reward_value_stack_config: ValueStackConfig = ValueStackConfig(20, [50])
                 ) -> None:
        """
        Initialize the StackObservationsActionRewardsWrapper.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        reward_scale : float, optional
            Scaling factor for rewards (default is 1.0).
        observation_value_stack_config : ValueStackConfig, optional
            Configuration for observation stacking (default is ValueStackConfig(1, [5])).
        action_value_stack_config : ValueStackConfig, optional
            Configuration for action stacking (default is ValueStackConfig(10, [20])).
        reward_value_stack_config : ValueStackConfig, optional
            Configuration for reward stacking (default is ValueStackConfig(20, [50])).
        """
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Only Box observation space handled"
        assert isinstance(env.action_space, Discrete), "Only Discrete action space handled"
        
        self.observation_value_stack = ValueStack(
            env.observation_space.shape, observation_value_stack_config.n_last_values,
            observation_value_stack_config.n_average_prev_last_values)
        self.action_value_stack = ValueStack(
            (env.action_space.n,), action_value_stack_config.n_last_values,
            action_value_stack_config.n_average_prev_last_values)
        self.reward_value_stack = ValueStack(
            (1,), reward_value_stack_config.n_last_values, reward_value_stack_config.n_average_prev_last_values
        )
        
        self.observation_space = Box(-1.0, 3.0, (self.observation_value_stack.n_return_elements +
                                                 self.action_value_stack.n_return_elements +
                                                 self.reward_value_stack.n_return_elements,))
        self.reward_scale = reward_scale

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment and return stacked observation.

        Parameters
        ----------
        action : int
            The action to take.

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict[str, Any]]
            (stacked_observation, reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)
        self.observation_value_stack.add_value(obs)
        self.action_value_stack.add_value(np.identity(self.env.action_space.n)[action])
        self.reward_value_stack.add_value(np.array([reward * self.reward_scale]))
        stacked_observation = self.get_stacked_observation()
        return stacked_observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        """
        Reset the environment and return stacked observation.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the environment reset.

        Returns
        -------
        np.ndarray
            The stacked observation.
        """
        self.observation_value_stack.reset()
        self.action_value_stack.reset()
        self.reward_value_stack.reset()
        obs = self.env.reset(**kwargs)
        self.observation_value_stack.add_value(obs)
        return self.get_stacked_observation()

    def get_stacked_observation(self) -> np.ndarray:
        """
        Get the concatenated stacked observation.

        Returns
        -------
        np.ndarray
            Flattened concatenation of observation, action, and reward stacks.
        """
        return np.concatenate([
            self.observation_value_stack.get_values().flatten(), 
            self.action_value_stack.get_values().flatten(),
            self.reward_value_stack.get_values().flatten()
        ])

