from typing import Type, Optional, Union

import gym
from stable_baselines3.common.vec_env import DummyVecEnv


def unwrap_wrapper_or_env(env: gym.Env, wrapper_class: Type[Union[gym.Wrapper, gym.Env]]) \
        -> Optional[Union[gym.Wrapper, gym.Env]]:
    """
    Retrieve a Wrapper or Env object by recursively searching through the environment chain.
    
    This function traverses the environment wrapper chain to find a specific wrapper
    or environment class. It handles both regular gym environments and DummyVecEnv.

    Parameters
    ----------
    env : gym.Env
        Environment to unwrap.
    wrapper_class : Type[Union[gym.Wrapper, gym.Env]]
        Wrapper or Env class to look for.
        
    Returns
    -------
    Optional[Union[gym.Wrapper, gym.Env]]
        Environment unwrapped till ``wrapper_class`` if it has been wrapped with it,
        None otherwise.
    """
    env_tmp = env
    
    # Handle DummyVecEnv by getting the first environment
    if isinstance(env_tmp, DummyVecEnv):
        env_tmp = env_tmp.envs[0]
    
    # Recursively traverse the wrapper chain
    while isinstance(env_tmp, gym.Wrapper) or isinstance(env_tmp, gym.Env):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    
    return None
