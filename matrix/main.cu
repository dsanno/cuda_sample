#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 32

typedef unsigned int uint;

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int main(int argc, char** argv){
	uint matrixSize = sizeof(uint)* MATRIX_SIZE * MATRIX_SIZE;

	int* hMatrixA;
	int* hMatrixB;
	int* hMatrixC;
	hMatrixA = (int*)malloc(matrixSize);
	hMatrixB = (int*)malloc(matrixSize);

	/* Matrix�̏����l�ݒ� */
	uint col_idx, row_idx;
	for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++){
		for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++){
			hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
			hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024 * 1024);
		}
	}

	/* �f�o�C�X���̕ϐ��ݒ� */
	int* dMatrixA;
	int* dMatrixB;
	int* dMatrixC;

	/* �f�o�C�X�������̈�̊m�� */
	cudaMalloc((void**)&dMatrixA, matrixSize);
	cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dMatrixB, matrixSize);
	cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dMatrixC, matrixSize);

	/* �u���b�N�T�C�Y�ƃO���b�h�T�C�Y�̐ݒ� */
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(MATRIX_SIZE / BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);

	cudaError_t error;
	/* �^�C�}�[���쐬���Čv���J�n */
	cudaEvent_t start;
	error = cudaEventCreate(&start);
	if (error != cudaSuccess) {
		printf("failed to craete start event");
		exit(EXIT_FAILURE);
	}
	cudaEvent_t stop;
	error = cudaEventCreate(&stop);
	if (error != cudaSuccess) {
		printf("failed to crete stop event");
		exit(EXIT_FAILURE);
	}

	error = cudaEventRecord(start, NULL);
	if (error != cudaSuccess) {
		printf("failed to record start event");
		exit(EXIT_FAILURE);
	}

	/* �J�[�l���̋N�� */
	matrixMul << <grid, block >> >(dMatrixA, dMatrixB, dMatrixC);
	cudaThreadSynchronize();

	/* ���ʂ̗̈�m�ۂƃf�o�C�X������̃������]�� */
	hMatrixC = (int*)malloc(matrixSize);
	cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost);

	error = cudaEventRecord(stop, NULL);
	if (error != cudaSuccess) {
		printf("failed to record stop event");
		exit(EXIT_FAILURE);
	}

	error = cudaEventSynchronize(stop);
	if (error != cudaSuccess) {
		printf("failed to synchronize");
		exit(EXIT_FAILURE);
	}

	/* �^�C�}�[���~ */
	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	if (error != cudaSuccess) {
		printf("failed to get elapsed time");
		exit(EXIT_FAILURE);
	}
	printf("Processing time: %f (msec)\n", msecTotal);
	printf("%d, %d\n", hMatrixC[0], hMatrixC[MATRIX_SIZE * MATRIX_SIZE - 1]);

	/* �z�X�g�E�f�o�C�X�������̊J�� */
	free(hMatrixA);
	free(hMatrixB);
	free(hMatrixC);
	cudaFree(dMatrixA);
	cudaFree(dMatrixB);
	cudaFree(dMatrixC);

	/* �I������ */
	cudaThreadExit();
}

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC){
	uint col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint row_idx = blockIdx.y * blockDim.y + threadIdx.y;
	uint scan_idx;
	uint target = 0;

	/*�s��̉��Z���s��*/
	for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
		target += inMatrixA[col_idx * MATRIX_SIZE + scan_idx] * inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
	}
	
	inMatrixC[col_idx * MATRIX_SIZE + row_idx] = target;
}
