from abc import ABC, abstractmethod
from typing import Optional


class RewardShaper(ABC):
    """
    Abstract base class for reward shaping in the environment.

    Attributes
    ----------
    env : Optional[PlannedActionEnv]
        The environment instance to which this shaper is attached.
    """
    def __init__(self) -> None:
        """
        Initialize the RewardShaper.
        """
        from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv
        self.env: Optional[PlannedActionEnv] = None

    def set_env(self, env) -> None:
        """
        Set the environment for the reward shaper.

        Parameters
        ----------
        env : PlannedActionEnv
            The environment instance.
        """
        from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv
        self.env: PlannedActionEnv = env

    @abstractmethod
    def get_shaped_reward(self) -> float:
        """
        Compute the shaped reward for the current step.

        Returns
        -------
        float
            The shaped reward value.
        """
        pass

    def reset(self) -> None:
        """
        Reset the reward shaper state at the beginning of an episode.
        """
        pass
