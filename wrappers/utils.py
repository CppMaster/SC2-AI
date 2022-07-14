from typing import Type, Optional, Union

import gym


def unwrap_wrapper_or_env(env: gym.Env, wrapper_class: Type[Union[gym.Wrapper, gym.Env]]) \
        -> Optional[Union[gym.Wrapper, gym.Env]]:
    """
    Retrieve a Wrapper or Env object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper or Env class to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper) or isinstance(env_tmp, gym.Env):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None
