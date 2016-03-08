all:
	gcc complex-matmul-harness.c -fopenmp -O3 -msse4 -o cmh -fassociative-math -ftree-vectorizer-verbose=2
