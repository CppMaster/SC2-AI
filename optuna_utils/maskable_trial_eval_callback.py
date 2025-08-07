from typing import Optional

import gym
import optuna
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback


class MaskableTrialEvalCallback(MaskableEvalCallback):
    """
    Callback used for evaluating and reporting a trial with action masking support.
    
    This callback extends MaskableEvalCallback to work with Optuna trials,
    allowing for trial pruning and reporting of evaluation results
    to the Optuna study while supporting action masking.

    Attributes
    ----------
    trial : optuna.Trial
        The Optuna trial associated with this evaluation.
    eval_idx : int
        Current evaluation index for reporting to Optuna.
    is_pruned : bool
        Whether the trial has been pruned.
    """
    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
            best_model_save_path: Optional[str] = None,
            log_path: Optional[str] = None,
            callback_after_eval: Optional[BaseCallback] = None
    ) -> None:
        """
        Initialize the MaskableTrialEvalCallback.

        Parameters
        ----------
        eval_env : gym.Env
            The environment used for evaluation.
        trial : optuna.Trial
            The Optuna trial for this training run.
        n_eval_episodes : int, optional
            Number of episodes to evaluate (default is 5).
        eval_freq : int, optional
            Evaluate every `eval_freq` timesteps (default is 10000).
        deterministic : bool, optional
            Whether to use deterministic actions during evaluation (default is True).
        verbose : int, optional
            Verbosity level (default is 0).
        best_model_save_path : Optional[str], optional
            Path to save the best model (default is None).
        log_path : Optional[str], optional
            Path to save evaluation logs (default is None).
        callback_after_eval : Optional[BaseCallback], optional
            Additional callback to run after evaluation (default is None).
        """
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            callback_after_eval=callback_after_eval
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        Called at each training step to perform evaluation and report to Optuna.

        Returns
        -------
        bool
            True to continue training, False to stop.
        """
        continue_training = True
        
        # Perform evaluation at specified frequency
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            continue_training = super()._on_step()
            self.eval_idx += 1
            
            # Report the best mean reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
                
        return continue_training
