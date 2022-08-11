from typing import List, Optional

import numpy as np
import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from wrappers.utils import unwrap_wrapper_or_env


class LogEpisodeCallback(BaseCallback):

    def __init__(self, mean_episodes: Optional[List[int]] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.mean_episodes = mean_episodes if mean_episodes is not None else []
        self.episodes_reported = 0

    def _init_callback(self) -> None:
        self.writer = tf.summary.create_file_writer(self.model.logger.dir)
        monitor = unwrap_wrapper_or_env(self.model.env, Monitor)
        assert monitor, "Monitor not found!"
        if isinstance(monitor, Monitor):
            self.monitor: Monitor = monitor
        else:
            raise RuntimeError("monitor not has wrong class")

    def _on_step(self) -> bool:
        if len(self.monitor.episode_returns) > self.episodes_reported:
            with self.writer.as_default(len(self.monitor.episode_returns) - 1):
                tf.summary.scalar("rollout/ep_rew", self.monitor.episode_returns[-1])
                tf.summary.scalar("rollout/ep_len", self.monitor.episode_lengths[-1])
                for mean in self.mean_episodes:
                    if len(self.monitor.episode_returns) >= mean:
                        tf.summary.scalar(f"rollout/ep_rew_mean_{mean}", np.mean(self.monitor.episode_returns[-mean:]))
                        tf.summary.scalar(f"rollout/ep_len_mean_{mean}", np.mean(self.monitor.episode_lengths[-mean:]))
                self.writer.flush()
            self.episodes_reported = len(self.monitor.episode_returns)

        return True
