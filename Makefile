all:
	maturin develop --release

debug:
	maturin develop

test: all
	python -m unittest test/test_robust_mean.py

test-debug: debug
	python -m unittest test/test_robust_mean.py