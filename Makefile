all:
	maturin develop --release

test: all
	python test/bench_means.py
