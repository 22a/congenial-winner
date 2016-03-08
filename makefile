all:
	gcc complex-matmul-harness.c -fopenmp -O3 -msse4 -o cmh -falign-functions=16 -falign-loops=16 -fassociative-math -ftree-vectorizer-verbose=2
