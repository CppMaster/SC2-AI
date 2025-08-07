from typing import List, Optional

import numpy as np
import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from wrappers.utils import unwrap_wrapper_or_env


class LogEpisodeCallback(BaseCallback):
    """
    Callback for logging episode information to TensorBoard.
    
    This callback logs episode rewards, lengths, and their moving averages
    to TensorBoard for monitoring training progress.

    Attributes
    ----------
    mean_episodes : List[int]
        List of episode counts for calculating moving averages.
    episodes_reported : int
        Number of episodes already reported to TensorBoard.
    writer : tf.summary.SummaryWriter
        TensorBoard writer for logging summaries.
    monitor : Monitor
        Monitor wrapper for accessing episode statistics.
    """
    def __init__(self, mean_episodes: Optional[List[int]] = None, verbose: int = 0) -> None:
        """
        Initialize the LogEpisodeCallback.

        Parameters
        ----------
        mean_episodes : List[int] or None, optional
            List of episode counts for calculating moving averages (default is None).
        verbose : int, optional
            Verbosity level (default is 0).
        """
        super().__init__(verbose=verbose)
        self.mean_episodes = mean_episodes if mean_episodes is not None else []
        self.episodes_reported = 0

    def _init_callback(self) -> None:
        """
        Initialize the callback with TensorBoard writer and monitor.
        """
        self.writer = tf.summary.create_file_writer(self.model.logger.dir)
        monitor = unwrap_wrapper_or_env(self.model.env, Monitor)
        assert monitor, "Monitor not found!"
        if isinstance(monitor, Monitor):
            self.monitor: Monitor = monitor
        else:
            raise RuntimeError("monitor not has wrong class")

    def _on_step(self) -> bool:
        """
        Called at each training step to log episode information.

        Returns
        -------
        bool
            Always returns True to continue training.
        """
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
