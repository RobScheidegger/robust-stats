import os

N = 10**4
D = 10**4

# Use these to ensure fairness, since this is all single-threaded
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import basic_mean_benchmark

INPUT_DATA = np.random.rand(N, D).astype(np.float32)
INPUT_DATA_TRANSPOSED = INPUT_DATA.T.copy()


def mean_numpy() -> np.ndarray:
    return np.mean(INPUT_DATA.T, axis=1)


def mean_rust_numpy() -> np.ndarray:
    return basic_mean_benchmark.mean_numpy(INPUT_DATA)


def mean_rust_native() -> np.ndarray:
    return basic_mean_benchmark.mean_native(INPUT_DATA)


def mean_rust_native_fast() -> np.ndarray:
    return basic_mean_benchmark.mean_native_fast(INPUT_DATA)


def mean_rust_native_blas() -> np.ndarray:
    return basic_mean_benchmark.mean_native_blas(INPUT_DATA)


__benchmarks__ = [
    # (e, mean_naive, "Naive (Python) - Baseline"),
    (mean_numpy, mean_numpy, "Numpy"),
    (mean_numpy, mean_rust_numpy, "Numpy (Rust)"),
    (mean_numpy, mean_rust_native, "Native (Rust)"),
    (mean_numpy, mean_rust_native_fast, "Native (Rust) - Alternate"),
    (mean_numpy, mean_rust_native_blas, "Native (Rust w/BLAS)"),
]

if __name__ == "__main__":
    print("Testing correctness of implementations...")

    naive = mean_numpy()
    native_transposed = np.mean(INPUT_DATA_TRANSPOSED, axis=1)

    for _, f, _ in __benchmarks__:
        print("Testing", f.__name__, "...", "OK" if np.allclose(naive, f()) else "FAIL")
