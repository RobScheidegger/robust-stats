"""
Contains tests of the basic functti
"""

import unittest
from robust_stats import robust_mean
from test.test_utils import create_test_data
import numpy as np


class TestRobustMeanMethods(unittest.TestCase):
    """
    Tests that the 'method' option of robust_mean works as expected, calling the respective implementation functions.
    """

    def test_uses_heuristic(self):
        test_data = create_test_data(100, 10)
        result = robust_mean(test_data, 0.1, method="heuristic")
        self.assertEqual(result.shape, (10,))

    def test_uses_filter(self):
        test_data = create_test_data(100, 10)
        result = robust_mean(test_data, 0.1, method="filter")
        self.assertEqual(result.shape, (10,))

    def test_uses_pgd(self):
        test_data = create_test_data(100, 10)
        result = robust_mean(test_data, 0.1, method="pgd")
        self.assertEqual(result.shape, (10,))


class TestRobustMeanLoss(unittest.TestCase):
    """
    Tests that on an un-corrupted dataset, the robust mean estimates are
    reasonably close to the actual mean.

    Note that there is a possible detrimental effect of doing this (since
    the dataset is non-corrupted), but even if it is not exact, it should be
    "reasonable".
    """

    def test_loss_heuristic(self):
        test_data = create_test_data(100, 10)
        result = robust_mean(test_data, 0.1, method="heuristic")
        actual = np.mean(test_data, axis=0)
        self.assertEqual(result.shape, (10,))
        difference = actual - result
        self.assertTrue(np.linalg.norm(difference) < 0.05)

    def test_loss_filter(self):
        test_data = create_test_data(100, 10)
        result = robust_mean(test_data, 0.1, method="filter")
        actual = np.mean(test_data, axis=0)
        self.assertEqual(result.shape, (10,))
        difference = actual - result
        self.assertTrue(np.linalg.norm(difference) < 0.05)

    def test_loss_pgd(self):
        test_data = create_test_data(100, 10)
        result = robust_mean(test_data, 0.1, method="pgd")
        actual = np.mean(test_data, axis=0)
        self.assertEqual(result.shape, (10,))
        difference = actual - result
        self.assertTrue(np.linalg.norm(difference) < 0.05)


if __name__ == "__main__":
    unittest.main()
