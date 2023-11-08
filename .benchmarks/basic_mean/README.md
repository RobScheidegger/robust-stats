# Benchmark: Basic Mean

This contains a simple benchmark to compare the trivial average computation in different forms. The idea is that in order to justify our use of Rust to make more complex forms of mean calculation efficient, we also want to demonstrate that for $n \times d$ arrays, we can perform the mean computation _at least as fast as_ the `numpy` implementation (Python standard) using Rust (including the Python bindings!).

Each mean is done of a series of $d$-vectors ($n$ of them), where for the experiments below we chose $d = n = 10^4$. All of the benchmarks are listed in terms of their relative performance to the regular numpy computation (called from Python).

|                 Benchmark | Min     | Max     | Mean    | Min (+)         | Max (+)         | Mean (+)        |
|---------------------------|---------|---------|---------|-----------------|-----------------|-----------------|
|                     Numpy | 0.191   | 0.197   | 0.192   | 0.191 (-1.0x)   | 0.194 (1.0x)    | 0.192 (1.0x)    |
|              Numpy (Rust) | 0.191   | 0.195   | 0.192   | 0.185 (1.0x)    | 0.187 (1.0x)    | 0.186 (1.0x)    |
|             Native (Rust) | 0.192   | 0.193   | 0.192   | 0.179 (1.1x)    | 0.183 (1.1x)    | 0.180 (1.1x)    |
| Native (Rust) - Alternate | 0.192   | 0.194   | 0.193   | 0.179 (1.1x)    | 0.181 (1.1x)    | 0.180 (1.1x)    |
|      Native (Rust w/BLAS) | 0.191   | 0.192   | 0.192   | 0.342 (-1.8x)   | 0.345 (-1.8x)   | 0.344 (-1.8x)   |

The idea here is to show that a simple, native Rust algorithm can actually outperform even the most optimized Python one, even when using Python bindings! We see that this is the case for the standard case of mean calculations, where we see around 10% improvement over the traditional python case.

Funnily enough, there is also a slight speedup from the traditional python case by just calling the `numpy` C-API from Rust instead of Python (second row in the table).