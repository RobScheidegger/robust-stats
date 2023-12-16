"""
Contains tests of the basic functti
"""

import unittest
from robust_stats import robust_mean
from test.test_utils import create_test_data


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


if __name__ == "__main__":
    unittest.main()
