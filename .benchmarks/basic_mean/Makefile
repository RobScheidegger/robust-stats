all:
	maturin develop --release

test: all
	richbench . --markdown --repeat 10 --times 10