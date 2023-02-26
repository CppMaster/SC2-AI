import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer


class DoubleActionRolloutBuffer(RolloutBuffer):

    def reset(self) -> None:
        super().reset()
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
