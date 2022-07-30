import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from wrappers.utils import unwrap_wrapper_or_env


class StopTrainingOnNoModelTrainingImprovement(BaseCallback):

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0,
                 mean_over_n_episodes: int = 100, eval_every_n_step: int = 10000):
        super(StopTrainingOnNoModelTrainingImprovement, self).__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.mean_over_n_episodes = mean_over_n_episodes
        self.eval_every_n_step = eval_every_n_step
        self.last_best_mean_reward = -np.inf
        self.best_mean_reward = -np.inf
        self.no_improvement_evals = 0
        self.n_evals = 0

    def _on_step(self) -> bool:
        monitor = unwrap_wrapper_or_env(self.training_env, Monitor)
        if not isinstance(monitor, Monitor):
            raise RuntimeError("StopTrainingOnNoModelTrainingImprovement needs Monitor wrapper")

        if self.n_calls % self.eval_every_n_step:
            return True

        continue_training = True
        if self.n_evals >= self.min_evals:
            mean_reward = np.mean(monitor.episode_returns[-self.mean_over_n_episodes:])
            logging.debug(f"Mean reward: {mean_reward}")
            self.best_mean_reward = max(mean_reward, self.best_mean_reward)
            logging.debug(f"Best mean reward: {self.best_mean_reward}")
            if self.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False
            logging.debug(f"No improvement evals: {self.no_improvement_evals}")

        self.last_best_mean_reward = self.best_mean_reward
        self.n_evals += 1

        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations. "
                f"Total timesteps: {self.num_timesteps}"
            )

        return continue_training
