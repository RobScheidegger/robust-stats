import numpy as np


def create_test_data(N: int, D: int) -> np.ndarray:
    """
    Creates an `n` x `d` array of gaussian data.

    Args:
        N (int): The number of samples.
        D (int): The dimension of each sample.

    Returns:
        np.ndarray: A random array containing gaussian data.
    """
    return np.random.rand(N, D).astype(np.float32)
