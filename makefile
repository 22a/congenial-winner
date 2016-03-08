all:
	gcc complex-matmul-harness.c -fopenmp -O3 -msse4 -o cmh -ftree-vectorizer-verbose=3 -fno-math-errno -fno-trapping-math -funroll-all-loops
