import numpy as np
import basic_mean_benchmark

N = 10**3
D = 10**5

ITERATIONS = 10**0

INPUT_DATA = np.random.rand(N, D).astype(np.float32)


def mean_numpy() -> np.ndarray:
    return np.mean(INPUT_DATA, axis=0)


def mean_rust_numpy() -> np.ndarray:
    return basic_mean_benchmark.mean_numpy(INPUT_DATA)


__benchmarks__ = [
    # (e, mean_naive, "Naive (Python) - Baseline"),
    (mean_numpy, mean_numpy, "Numpy"),
    (mean_numpy, mean_rust_numpy, "Numpy (Rust)"),
]
