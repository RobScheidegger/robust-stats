import numpy as np

N = 100000
D = 1000

ITERATIONS = 10**4


def benchmark():
    return np.mean(np.random.rand(1000))


__benchmarks__ = []
