# Benchmark: Basic Mean

This contains a simple benchmark to compare the trivial average computation in different forms:

1. Python (Naive) - Ignored for the sake of tests, because this is several orders of magnitude slower.
2. Python (Numpy) 
3. Python-wrapped Naive Rust Implementation
4. Python-wrapped BLAS Rust Implementation
5. Python-wrapped Rust Numpy 
5. Python-wrapped cuBLAS Rust Implementation

Each mean is done of a series of $d$-vectors ($n$ of them). 