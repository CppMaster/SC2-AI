import gym


class RewardScaleWrapper(gym.RewardWrapper):
    """
    Wrapper to scale rewards by a constant factor.
    
    This wrapper multiplies all rewards by a specified scale factor,
    which can be useful for normalizing rewards or adjusting their magnitude.

    Attributes
    ----------
    scale : float
        The scaling factor to apply to rewards.
    """
    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        """
        Initialize the RewardScaleWrapper.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        scale : float, optional
            The scaling factor for rewards (default is 1.0).
        """
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        """
        Scale the reward by the specified factor.

        Parameters
        ----------
        reward : float
            The original reward.

        Returns
        -------
        float
            The scaled reward.
        """
        return reward * self.scale
