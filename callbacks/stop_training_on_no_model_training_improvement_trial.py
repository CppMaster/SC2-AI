import optuna

from callbacks.stop_training_on_no_model_training_improvement import StopTrainingOnNoModelTrainingImprovement


class StopTrainingOnNoModelTrainingImprovementTrial(StopTrainingOnNoModelTrainingImprovement):
    """
    Callback to stop training when no improvement is observed, with Optuna trial integration.
    
    This callback extends StopTrainingOnNoModelTrainingImprovement to work with Optuna trials,
    allowing for trial pruning and reporting of results to the Optuna study.

    Attributes
    ----------
    trial : optuna.Trial
        The Optuna trial associated with this training run.
    is_pruned : bool
        Whether the trial has been pruned.
    """
    def __init__(self, max_no_improvement_evals: int, trail: optuna.Trial, min_evals: int = 0, verbose: int = 0,
                 mean_over_n_episodes: int = 100, eval_every_n_step: int = 10000) -> None:
        """
        Initialize the StopTrainingOnNoModelTrainingImprovementTrial callback.

        Parameters
        ----------
        max_no_improvement_evals : int
            Maximum number of evaluations without improvement before stopping.
        trail : optuna.Trial
            The Optuna trial for this training run.
        min_evals : int, optional
            Minimum number of evaluations before checking for improvement (default is 0).
        verbose : int, optional
            Verbosity level (default is 0).
        mean_over_n_episodes : int, optional
            Number of episodes to average over for reward calculation (default is 100).
        eval_every_n_step : int, optional
            Number of steps between evaluations (default is 10000).
        """
        super(StopTrainingOnNoModelTrainingImprovementTrial, self).__init__(
            max_no_improvement_evals, min_evals, verbose, mean_over_n_episodes, eval_every_n_step
        )
        self.trial = trail
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        Called at each training step to check for improvement and handle trial pruning.

        Returns
        -------
        bool
            True to continue training, False to stop.
        """
        # Only evaluate every eval_every_n_step steps
        if self.n_calls % self.eval_every_n_step:
            return True

        continue_training = super()._on_step()
        self.trial.report(self.last_best_mean_reward, self.n_evals)
        
        # Prune trial if needed
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return continue_training
