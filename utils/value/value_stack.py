from typing import List, Tuple, Optional

import numpy as np


class ValueStack:
    """
    A stack for storing and retrieving values with averaging capabilities.
    
    This class maintains a stack of values and provides functionality to retrieve
    recent values and averages of previous values. Useful for creating temporal
    features in reinforcement learning environments.

    Attributes
    ----------
    value_shape : Tuple[int, ...]
        Shape of individual values in the stack.
    n_last_values : int
        Number of most recent values to keep.
    n_average_prev_last_values : List[int]
        List of window sizes for averaging previous values.
    values : List[np.ndarray]
        List of stored values.
    max_size : int
        Maximum number of values to store.
    zero_value : np.ndarray
        Zero value with the same shape as input values.
    result_shape : Tuple[int, ...]
        Shape of the result array.
    n_return_elements : int
        Total number of elements in the result array.
    """
    def __init__(self, value_shape: Tuple[int, ...], n_last_values: int = 1,
                 n_average_prev_last_values: Optional[List[int]] = None) -> None:
        """
        Initialize the ValueStack.

        Parameters
        ----------
        value_shape : Tuple[int, ...]
            Shape of individual values to be stored.
        n_last_values : int, optional
            Number of most recent values to keep (default is 1).
        n_average_prev_last_values : Optional[List[int]], optional
            List of window sizes for averaging previous values (default is None).
        """
        self.value_shape = value_shape
        self.n_last_values = n_last_values
        self.n_average_prev_last_values = n_average_prev_last_values or []
        self.values: List[np.ndarray] = []
        self.max_size = n_last_values + sum(self.n_average_prev_last_values)
        self.zero_value = np.zeros(self.value_shape)
        self.result_shape = (n_last_values + len(self.n_average_prev_last_values),) + self.value_shape
        self.n_return_elements = np.prod(self.result_shape)

    def add_value(self, value: np.ndarray) -> None:
        """
        Add a value to the stack.

        Parameters
        ----------
        value : np.ndarray
            The value to add to the stack.

        Raises
        ------
        AssertionError
            If the value shape doesn't match the expected shape.
        """
        assert value.shape == self.value_shape, f"Wrong value shape. Expected: {self.value_shape}, got: {value.shape}"
        self.values.append(value)
        self.values = self.values[-self.max_size:]

    def reset(self) -> None:
        """
        Reset the stack by clearing all stored values.
        """
        self.values: List[np.ndarray] = []

    def get_values(self) -> np.ndarray:
        """
        Get the stacked values including recent values and averages.

        Returns
        -------
        np.ndarray
            Array containing the most recent values followed by averages
            of previous values according to the configuration.
        """
        # Get the most recent values in reverse order
        result: List[np.ndarray] = self.values[:-self.n_last_values-1:-1]
        result.extend([self.zero_value] * max(self.n_last_values-len(result), 0))
        
        # Calculate averages of previous values
        index = self.n_last_values
        for n_prev in self.n_average_prev_last_values:
            values_to_average = self.values[-index-1:-index-n_prev-1:-1]
            result.append(np.mean(values_to_average, axis=0) if values_to_average else self.zero_value)
            index += n_prev
        
        return np.array(result)
