"""
This module contains the main robust_mean entrypoint of the robust_stats package.
"""

from robust_stats.robust_stats import (
    robust_mean_heuristic,
    robust_mean_filter,
    robust_mean_pgd,
)
import numpy as np


def robust_mean(
    input: np.ndarray, epsilon: float, method: str = "heuristic"
) -> np.ndarray:
    """
    Compute the robust mean of the input array.

    Parameters
    ----------
    input : np.ndarray
        The input array to compute the robust mean of.
    epsilon : float
        The percentage of the input data that could be corrupted.
    method : str
        The method to use to compute the robust mean. Must be one of:
        - "heuristic"
        - "filter"
        - "pgd"

    Returns
    -------
    np.ndarray
        The robust mean of the input array.
    """
    if method == "heuristic":
        return robust_mean_heuristic(input, epsilon)
    elif method == "filter":
        return robust_mean_filter(input, epsilon)
    elif method == "pgd":
        return robust_mean_pgd(input, epsilon)
    else:
        raise ValueError(f"Invalid method {method}")
