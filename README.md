# robust-stats

Provides highly efficient implementations of various algorithm in robust statistics in Rust, with direct bindings to Python. The current focus is on robust mean estimation for known adversarial corruption for Gaussian distributions, but may later extend to robust covariance estimation or other useful robust statistical methods based on recent research efforts. 

As far as this package is concerned, being _robust_ means being able to compute the mean of a dataset where at most an $\epsilon$-fraction of the data is corrupted by an arbitrary adversary (can replace any set of the original sample with arbitrary other points, so long as it only affects an $\epsilon$-fraction of the input dataset).

## Robust Mean Estimation

The `robust_stats.robust_mean` function computes the mean of an $n$-datapoint sample of a $d$-dimensional _Gaussian_ dataset with identity covariance by default.

## Usage

The `robust_stats` package can be installed via `pip` as follows:

```bash
pip install robust_stats
```

To compute the robust mean of a 2-dimensional numpy array with some $\epsilon$ sample corruption, then you can use the package as follows:

```python
from robust_stats import robust_mean

x = get_data() # Returns some numpy array with dimension (n, d)
x_bar = robust_mean(x, 0.1, method='heuristic')
```

The arguments to `robust_mean` are as follows:

1. `x: np.ndarray` - the $(n \times d)$ input array to compute the mean of.
2. `epsilon: float` - the percent of the data that is thought to be corrupted, in $[0, 1]$.
3. `method: str = 'heuristic'` - which robust mean estimation method to use. Must be one of the methods defined below. Defaults to `heuristic`.

It returns a $d$-dimensional output `np.ndarray` containing the computed mean value.

## Methods

We provide efficient implementations of several means of robust mean estimation, based on relevant recent work in robust statistics. These methods 

### (`heuristic`)

Rust Source: `src/robust/mean/robust_mean_heuristic.rs`

### (`filter`)

Rust Source: `src/robust/mean/robust_mean_filter.rs`

### Projected Gradient Descent (`pgd`)


Rust Source: `src/robust/mean/robust_mean_pgd.rs`

## Acknowledgements

This project was our final project for Brown's CSCI2952Q (Robust Algorithms in Machine Learning) taught by Professor Yu Cheng.

## Maintainers

- Robert Scheidegger (robert_scheidegger@brown.edu)
- Hammad Izhar (hammad_izhar@brown.edu)