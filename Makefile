all:
	maturin develop

test: all
	python -m unittest test/test_robust_mean.py