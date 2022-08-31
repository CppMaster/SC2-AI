from typing import List, Tuple, Optional

import numpy as np


class ValueStack:

    def __init__(self, value_shape: Tuple[int, ...], n_last_values: int = 1,
                 n_average_prev_last_values: Optional[List[int]] = None):
        self.value_shape = value_shape
        self.n_last_values = n_last_values
        self.n_average_prev_last_values = n_average_prev_last_values or []
        self.values: List[np.ndarray] = []
        self.max_size = n_last_values + sum(self.n_average_prev_last_values)
        self.zero_value = np.zeros(self.value_shape)
        self.result_shape = (n_last_values + len(n_average_prev_last_values),) + self.value_shape
        self.n_return_elements = np.prod(self.result_shape)

    def add_value(self, value: np.ndarray):
        assert value.shape == self.value_shape, f"Wrong value shape. Expected: {self.value_shape}, got: {value.shape}"
        self.values.append(value)
        self.values = self.values[-self.max_size:]

    def reset(self):
        self.values: List[np.ndarray] = []

    def get_values(self) -> np.ndarray:
        result: List[np.ndarray] = self.values[:-self.n_last_values-1:-1]
        result.extend([self.zero_value] * max(self.n_last_values-len(result), 0))
        index = self.n_last_values
        for n_prev in self.n_average_prev_last_values:
            values_to_average = self.values[-index-1:-index-n_prev-1:-1]
            result.append(np.mean(values_to_average, axis=0) if values_to_average else self.zero_value)
            index += n_prev
        return np.array(result)
