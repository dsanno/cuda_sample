#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1024

int main(int argc, char **argv)
{
	int *ma, *mb, *mc;
	int i, j, k;
	int x;
	clock_t start, stop;

	ma = malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	mb = malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	mc = malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
	for (j = 0; j < 10; j++) {
		for (i = 0; i < MATRIX_SIZE; i++) {
			ma[j * MATRIX_SIZE + i] = rand() % 1024;
			mb[j * MATRIX_SIZE + i] = rand() % 1024;
		}
	}

	start = clock();

	for (j = 0; j < MATRIX_SIZE; j++) {
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (k = 0; k < MATRIX_SIZE; k++) {
				mc[j * MATRIX_SIZE + i] += ma[j * MATRIX_SIZE + k] * mb[k * MATRIX_SIZE + i];
			}
		}
	}

	stop = clock();

	printf("%d ms\n", stop - start);

	free(ma);
	free(mb);
	free(mc);
}
