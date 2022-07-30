import optuna

from callbacks.stop_training_on_no_model_training_improvement import StopTrainingOnNoModelTrainingImprovement


class StopTrainingOnNoModelTrainingImprovementTrial(StopTrainingOnNoModelTrainingImprovement):

    def __init__(self, max_no_improvement_evals: int, trail: optuna.Trial, min_evals: int = 0, verbose: int = 0,
                 mean_over_n_episodes: int = 100, eval_every_n_step: int = 10000):
        super(StopTrainingOnNoModelTrainingImprovementTrial, self).__init__(
            max_no_improvement_evals, min_evals, verbose, mean_over_n_episodes, eval_every_n_step
        )
        self.trial = trail
        self.is_pruned = False

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_every_n_step:
            return True

        continue_training = super()._on_step()
        self.trial.report(self.last_best_mean_reward, self.n_evals)
        # Prune trial if needed
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return continue_training
