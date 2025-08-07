import logging
from typing import List
import numpy as np

from pysc2.env.sc2_env import Difficulty


class DifficultyScheduler:
    """
    Scheduler for dynamically adjusting game difficulty based on performance.
    
    This class monitors the agent's performance and adjusts the difficulty
    of the opponent to maintain an appropriate challenge level.

    Attributes
    ----------
    current_difficulty : Difficulty
        The current difficulty level.
    min_mean_score : float
        Minimum mean score threshold for decreasing difficulty.
    max_mean_score : float
        Maximum mean score threshold for increasing difficulty.
    n_scores_mean : int
        Number of scores to use for calculating the mean.
    scores : List[float]
        List of recent scores for calculating the mean.
    logger : logging.Logger
        Logger for this scheduler.
    """
    def __init__(self, starting_difficulty: Difficulty = Difficulty.very_easy, min_mean_score: float = -.5,
                 max_mean_score: float = .5, n_scores_mean: int = 5) -> None:
        """
        Initialize the DifficultyScheduler.

        Parameters
        ----------
        starting_difficulty : Difficulty, optional
            Initial difficulty level (default is Difficulty.very_easy).
        min_mean_score : float, optional
            Minimum mean score threshold for decreasing difficulty (default is -0.5).
        max_mean_score : float, optional
            Maximum mean score threshold for increasing difficulty (default is 0.5).
        n_scores_mean : int, optional
            Number of scores to use for calculating the mean (default is 5).
        """
        self.current_difficulty = starting_difficulty
        self.min_mean_score = min_mean_score
        self.max_mean_score = max_mean_score
        self.n_scores_mean = n_scores_mean
        self.scores: List[float] = []
        self.logger = logging.getLogger("DifficultyScheduler")

    def report_score(self, score: float) -> Difficulty:
        """
        Report a score and potentially adjust difficulty.

        Parameters
        ----------
        score : float
            The score to report.

        Returns
        -------
        Difficulty
            The current difficulty level after potential adjustment.
        """
        self.scores.append(score)
        self.scores = self.scores[-self.n_scores_mean:]
        n_scores = len(self.scores)
        mean_scores = float(np.mean(self.scores))
        self.logger.debug(f"n_scores: {n_scores},\tmean_scores: {mean_scores}, difficulty: {self.current_difficulty}")

        # Don't adjust difficulty until we have enough scores
        if n_scores < self.n_scores_mean:
            return self.current_difficulty

        # Increase difficulty if performance is too good
        if mean_scores >= self.max_mean_score and self.current_difficulty < Difficulty.very_hard:
            self.current_difficulty = Difficulty(self.current_difficulty + 1)
            self.scores = []
            self.logger.info(f"Difficulty increased to: {self.current_difficulty}")
        # Decrease difficulty if performance is too poor
        elif mean_scores <= self.min_mean_score and self.current_difficulty > Difficulty.very_easy:
            self.current_difficulty = Difficulty(self.current_difficulty - 1)
            self.scores = []
            self.logger.info(f"Difficulty decreased to: {self.current_difficulty}")

        return self.current_difficulty



