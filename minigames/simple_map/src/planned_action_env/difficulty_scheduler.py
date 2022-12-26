import logging
from typing import List
import numpy as np

from pysc2.env.sc2_env import Difficulty


class DifficultyScheduler:

    def __init__(self, starting_difficulty: Difficulty = Difficulty.very_easy, min_mean_score: float = -.5,
                 max_mean_score: float = .5, n_scores_mean: int = 5):
        self.current_difficulty = starting_difficulty
        self.min_mean_score = min_mean_score
        self.max_mean_score = max_mean_score
        self.n_scores_mean = n_scores_mean
        self.scores: List[float] = []
        self.logger = logging.getLogger("DifficultyScheduler")

    def report_score(self, score: float) -> Difficulty:
        self.scores.append(score)
        self.scores = self.scores[-self.n_scores_mean:]
        n_scores = len(self.scores)
        mean_scores = float(np.mean(self.scores))
        self.logger.debug(f"n_scores: {n_scores},\tmean_scores: {mean_scores}, difficulty: {self.current_difficulty}")

        if n_scores < self.n_scores_mean:
            return self.current_difficulty

        if mean_scores >= self.max_mean_score and self.current_difficulty < Difficulty.very_hard:
            self.current_difficulty = Difficulty(self.current_difficulty + 1)
            self.scores = []
            self.logger.info(f"Difficulty increased to: {self.current_difficulty}")
        elif mean_scores <= self.min_mean_score and self.current_difficulty > Difficulty.very_easy:
            self.current_difficulty = Difficulty(self.current_difficulty - 1)
            self.scores = []
            self.logger.info(f"Difficulty decreased to: {self.current_difficulty}")

        return self.current_difficulty



