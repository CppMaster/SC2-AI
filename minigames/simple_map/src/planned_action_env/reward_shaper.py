from abc import ABC, abstractmethod
from typing import Optional


class RewardShaper(ABC):

    def __init__(self):
        from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv
        self.env: Optional[PlannedActionEnv] = None

    def set_env(self, env):
        from minigames.simple_map.src.planned_action_env.env import PlannedActionEnv
        self.env: PlannedActionEnv = env

    @abstractmethod
    def get_shaped_reward(self) -> float:
        pass

    def reset(self):
        pass
