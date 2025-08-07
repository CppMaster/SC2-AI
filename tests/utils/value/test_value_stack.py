from typing import Tuple, List, Optional

import numpy as np
import pytest

from utils.value.value_stack import ValueStack


class TestValueStack:
    """
    Test cases for the ValueStack class.
    
    This class contains comprehensive tests for the ValueStack functionality,
    including tests for different value shapes, stack sizes, and averaging
    configurations.
    """
    
    @pytest.mark.parametrize(("value_shape", "n_last_values", "n_average_prev_last_values", "values", "output"), [
        ((1, ), 1, [], [np.array([0])], np.array([[0]])),
        ((), 1, [], [np.array(0)], np.array([0])),
        ((1, 1), 1, [], [np.array([[0]])], np.array([[[0]]])),
        ((), 1, [], [np.array(0), np.array(1)], np.array([1])),
        ((), 1, [2], [np.array(0), np.array(1), np.array(2)], np.array([2, 0.5])),
        ((), 2, [2, 3], [np.array(0), np.array(1), np.array(2), np.array(3), np.array(4), np.array(5)],
         np.array([5, 4, 2.5, 0.5])),
        ((), 1, [2, 3, 4, 5], [np.array(0), np.array(1), np.array(2)], np.array([2, 0.5, 0, 0, 0])),
    ])
    def test_get_values(self, value_shape: Tuple[int, ...], n_last_values: int,
                        n_average_prev_last_values: Optional[List[int]], values: List[np.ndarray], output: np.ndarray) -> None:
        """
        Test the get_values method with various configurations.
        
        This test verifies that the ValueStack correctly handles different
        value shapes, stack sizes, and averaging configurations.

        Parameters
        ----------
        value_shape : Tuple[int, ...]
            Shape of the values to be stored.
        n_last_values : int
            Number of most recent values to keep.
        n_average_prev_last_values : Optional[List[int]]
            List of window sizes for averaging previous values.
        values : List[np.ndarray]
            List of values to add to the stack.
        output : np.ndarray
            Expected output from get_values().
        """
        value_stack = ValueStack(value_shape, n_last_values, n_average_prev_last_values)
        
        # Add all values to the stack
        for value in values:
            value_stack.add_value(value)
        
        # Get the result and compare with expected output
        result = value_stack.get_values()
        assert np.array_equal(result, output)
