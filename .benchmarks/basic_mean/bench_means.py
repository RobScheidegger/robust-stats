import os

# Use these to ensure fairness, since this is all single-threaded
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import basic_mean_benchmark
import time


def mean_numpy(input: np.ndarray) -> np.ndarray:
    return np.mean(input, axis=0)


def mean_rust_native(input: np.ndarray) -> np.ndarray:
    return basic_mean_benchmark.mean_native(input)


def mean_rust_numpy(input: np.ndarray) -> np.ndarray:
    return basic_mean_benchmark.mean_numpy(input)


__benchmarks__ = [
    # (e, mean_naive, "Naive (Python) - Baseline"),
    (mean_numpy, mean_numpy, "Numpy"),
    (mean_numpy, mean_rust_native, "Native (Rust)"),
]

if __name__ == "__main__":
    N = 10**4
    D = 10**5

    SAMPLES = 100

    input_data = np.random.rand(N, D).astype(np.float32)
    # Skip every other sample in the input array
    input_data = input_data[::2]

    # First, time the amount of time the regular numpy implementation takes
    start = time.time()
    for _ in range(SAMPLES):
        mean_numpy(input_data)
    end = time.time()

    print(f"Python (Numpy) took {end - start} seconds")

    # Now, time the amount of time the Rust implementation takes
    start = time.time()
    for _ in range(SAMPLES):
        mean_rust_native(input_data)
    end = time.time()

    print(f"Rust (Native) took {end - start} seconds")

    start = time.time()
    for _ in range(SAMPLES):
        mean_rust_numpy(input_data)
    end = time.time()

    print(f"Rust (Numpy) took {end - start} seconds")
