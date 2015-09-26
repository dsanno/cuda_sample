#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 224
#define MAX_STATIC_SIZE 4096
#define MAX_PROBLEM_NUM 1024
#define CELL_NUM 81
#define ROW_NUM 9
#define NEXT_NUMBER_NUM (1 << ROW_NUM)

static int load(char *in_file_path, int *out_number);
static int find_valid_number(const int *in_static_number, int in_depth, int in_max_count, int* out_result, int *, int *inout_initial);

__constant__ int static_count;
__constant__ int next_number[NEXT_NUMBER_NUM];
__global__ void solve_sudoku(int *in_static_number, int *out_result, int *out_count);

int main(int argc, char** argv){
	int host_static_number[MAX_PROBLEM_NUM * CELL_NUM];
	int result[MAX_PROBLEM_NUM * CELL_NUM];
	int valid_index;
	int initial[CELL_NUM];
	int answer_num[MAX_PROBLEM_NUM];
	int host_next_number[512];
	int problem_num;
	int *device_result;
	int *device_count;
	int count;
	int *device_static_number;
	int *valid_number;
	int i, j, k;

	if (argc < 2 || argc >= 3) {
		printf("Usage: sudoku_cpu file_path");
		return 1;
	}
	char *file_path = argv[1];
	problem_num = load(file_path, host_static_number);
	if (problem_num <= 0) {
		printf("Can't load file %s.", file_path);
		return 1;
	}

	cudaMalloc((void**)&device_result, sizeof(int) * CELL_NUM);
	cudaMalloc((void**)&device_count, sizeof(int));
	valid_number = (int*)malloc(sizeof(int) * CELL_NUM * MAX_STATIC_SIZE);
	cudaMalloc((void**)&device_static_number, sizeof(int) * CELL_NUM * MAX_STATIC_SIZE);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	cudaError_t error;
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

	for (i = 0; i < NEXT_NUMBER_NUM; i++) {
		for (j = 0; j < ROW_NUM + 1; j++) {
			if (((1 << j) & i) == 0) {
				host_next_number[i] = j;
				break;
			}
		}
	}
	valid_index = -1;
	for (i = 0; i < problem_num; i++) {
		answer_num[i] = 0;
		do {
			count = find_valid_number(&host_static_number[i * CELL_NUM], 16, MAX_STATIC_SIZE, valid_number, &valid_index, initial);
			cudaMemcpy(device_static_number, valid_number, sizeof(int)* CELL_NUM * count, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(static_count, &count, sizeof(int));
			cudaMemcpyToSymbol(next_number, host_next_number, sizeof(host_next_number));
			cudaMemset(device_result, 0, sizeof(int)* CELL_NUM);
			cudaMemset(device_count, 0, sizeof(int));

			dim3 block(BLOCK_SIZE, 1);
			dim3 grid(deviceProp.multiProcessorCount, 1);
			solve_sudoku << <grid, block >> >(device_static_number, device_result, device_count);
			cudaThreadSynchronize();

			cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost);
			if (answer_num[i] == 0 && count > 0) {
				cudaMemcpy(&result[i * CELL_NUM], device_result, sizeof(int)* CELL_NUM, cudaMemcpyDeviceToHost);
			}
			answer_num[i] += count;
		} while (valid_index >= 0);
	}

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

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	if (error != cudaSuccess) {
		printf("failed to get elapsed time");
		exit(EXIT_FAILURE);
	}
	printf("Processing time: %f (msec)\n", msecTotal);

	for (i = 0; i < problem_num; i++) {
		printf("%d found\n", answer_num[i]);
		for (j = 0; j < ROW_NUM; j++) {
			for (k = 0; k < ROW_NUM; k++) {
				printf("%d ", result[i * CELL_NUM + j * ROW_NUM + k]);
			}
			printf("\n");
		}
	}
	cudaFree(device_result);
	cudaFree(device_count);
	cudaFree(device_static_number);
	free(valid_number);

	cudaThreadExit();
}

static int
load(char *in_file_path, int *out_number) {
	char buf[1024];
	errno_t error;
	FILE *fp;
	int size;
	int i, n;

	error = fopen_s(&fp, in_file_path, "r");
	if (error != 0) {
		return 0;
	}
	size = fread(buf, 1, sizeof(buf), fp);
	fclose(fp);

	for (i = 0, n = 0; i < size && n < MAX_PROBLEM_NUM * CELL_NUM; i++) {
		if (buf[i] >= '1' && buf[i] <= '9') {
			out_number[n] = buf[i] - '0';
			n++;
		}
		else if (buf[i] == '-') {
			out_number[n] = 0;
			n++;
		}
	}
	return n / CELL_NUM;
}

static int
find_valid_number(const int *in_static_number, int in_depth, int in_max_count, int* out_result, int *inout_index, int *inout_initial) {
	const int row[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 2, 2,
		3, 3, 3, 3, 3, 3, 3, 3, 3,
		4, 4, 4, 4, 4, 4, 4, 4, 4,
		5, 5, 5, 5, 5, 5, 5, 5, 5,
		6, 6, 6, 6, 6, 6, 6, 6, 6,
		7, 7, 7, 7, 7, 7, 7, 7, 7,
		8, 8, 8, 8, 8, 8, 8, 8, 8
	};
	const int col[] = {
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8
	};
	const int box[] = {
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		0, 0, 0, 1, 1, 1, 2, 2, 2,
		3, 3, 3, 4, 4, 4, 5, 5, 5,
		3, 3, 3, 4, 4, 4, 5, 5, 5,
		3, 3, 3, 4, 4, 4, 5, 5, 5,
		6, 6, 6, 7, 7, 7, 8, 8, 8,
		6, 6, 6, 7, 7, 7, 8, 8, 8,
		6, 6, 6, 7, 7, 7, 8, 8, 8
	};
	int row_flag[ROW_NUM] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int col_flag[ROW_NUM] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int box_flag[ROW_NUM] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int empty[CELL_NUM];
	int number[CELL_NUM];
	int found = 0;
	int pos;
	int flag;
	int index;
	int empty_num;
	int i;
	int *result = out_result;

	empty_num = 0;
	for (pos = 0; pos < CELL_NUM; pos++) {
		if (*inout_index >= 0) {
			number[pos] = inout_initial[pos];
		} else {
			number[pos] = in_static_number[pos];
		}
		if (number[pos] > 0) {
			flag = 1 << number[pos];
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
		}
		if (in_static_number[pos] == 0) {
			empty[empty_num] = pos;
			empty_num++;
		}
	}
	if (*inout_index >= 0) {
		index = *inout_index;
		pos = empty[index];
		flag = 1 << number[pos];
		row_flag[row[pos]] ^= flag;
		col_flag[col[pos]] ^= flag;
		box_flag[box[pos]] ^= flag;
	} else {
		index = 0;
	}
	while (1) {
		pos = empty[index];
		for (number[pos]++; number[pos] < ROW_NUM + 1; number[pos]++) {
			flag = 1 << number[pos];
			if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
				continue;
			}
			if (index >= in_depth - 1 || index >= empty_num - 1) {
				for (i = 0; i < CELL_NUM; i++) {
					result[i] = number[i];
				}
				found++;
				result += CELL_NUM;
				if (found >= in_max_count) {
					break;
				}
			} else {
				break;
			}
		}
		if (found >= in_max_count) {
			break;
		}
		if (number[pos] < ROW_NUM + 1) {
			flag = 1 << number[pos];
			index++;
		} else {
			index--;
			if (index < 0) {
				break;
			}
			number[pos] = 0;
			pos = empty[index];
			flag = 1 << number[pos];
		}
		row_flag[row[pos]] ^= flag;
		col_flag[col[pos]] ^= flag;
		box_flag[box[pos]] ^= flag;
	}
	if (index >= 0) {
		for (pos = 0; pos < CELL_NUM; pos++) {
			inout_initial[pos] = number[pos];
		}
	}
	*inout_index = index;
	return found;
}

__constant__ int row[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 1, 1, 1, 1, 1, 1,
	2, 2, 2, 2, 2, 2, 2, 2, 2,
	3, 3, 3, 3, 3, 3, 3, 3, 3,
	4, 4, 4, 4, 4, 4, 4, 4, 4,
	5, 5, 5, 5, 5, 5, 5, 5, 5,
	6, 6, 6, 6, 6, 6, 6, 6, 6,
	7, 7, 7, 7, 7, 7, 7, 7, 7,
	8, 8, 8, 8, 8, 8, 8, 8, 8
};
__constant__ int col[] = {
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8,
	0, 1, 2, 3, 4, 5, 6, 7, 8
};
__constant__ int box[] = {
	0, 0, 0, 1, 1, 1, 2, 2, 2,
	0, 0, 0, 1, 1, 1, 2, 2, 2,
	0, 0, 0, 1, 1, 1, 2, 2, 2,
	3, 3, 3, 4, 4, 4, 5, 5, 5,
	3, 3, 3, 4, 4, 4, 5, 5, 5,
	3, 3, 3, 4, 4, 4, 5, 5, 5,
	6, 6, 6, 7, 7, 7, 8, 8, 8,
	6, 6, 6, 7, 7, 7, 8, 8, 8,
	6, 6, 6, 7, 7, 7, 8, 8, 8
};

__global__ void
solve_sudoku(int *in_static_number, int *out_result, int *out_count) {
	__shared__ int found;
	__shared__ short int row_flag[ROW_NUM][BLOCK_SIZE];
	__shared__ short int col_flag[ROW_NUM][BLOCK_SIZE];
	__shared__ short int box_flag[ROW_NUM][BLOCK_SIZE];
	__shared__ char number[CELL_NUM][BLOCK_SIZE];
	__shared__ char empty[CELL_NUM][BLOCK_SIZE];
	__shared__ int static_index;
	__shared__ int static_index_end;
	int pos;
	int index;
	int flag;
	int i;
	int offset = -1;

	if (threadIdx.x == 0) {
		found = 0;
		static_index = (static_count * blockIdx.x) / gridDim.x;
		static_index_end = (static_count * (blockIdx.x + 1)) / gridDim.x;
	}
	__syncthreads();
	index = -1;
	while (1) {
		if (index < 0) {
			i = atomicAdd(&static_index, 1);
			if (i >= static_index_end) {
				break;
			}
#pragma unroll
			for (pos = 0; pos < ROW_NUM; pos++) {
				row_flag[pos][threadIdx.x] = 0;
				col_flag[pos][threadIdx.x] = 0;
				box_flag[pos][threadIdx.x] = 0;
			}
			offset = i * CELL_NUM;
			i = 0;
#pragma unroll
			for (pos = 0; pos < CELL_NUM; pos++) {
				number[pos][threadIdx.x] = in_static_number[offset + pos];
				if (number[pos][threadIdx.x] > 0) {
					flag = 1 << number[pos][threadIdx.x];
					row_flag[row[pos]][threadIdx.x] |= flag;
					col_flag[col[pos]][threadIdx.x] |= flag;
					box_flag[box[pos]][threadIdx.x] |= flag;
				} else {
					empty[i][threadIdx.x] = pos;
					i++;
				}
			}
			empty[i][threadIdx.x] = -1;
			index = 0;
		}
		pos = empty[index][threadIdx.x];
		if (pos < 0) {
			int old_found = atomicAdd(&found, 1);
			if (old_found == 0) {
#pragma unroll
				for (int i = 0; i < CELL_NUM; i++) {
					out_result[i] = number[i][threadIdx.x];
				}
			}
			index--;
			pos = empty[index][threadIdx.x];
			flag = 1 << number[pos][threadIdx.x];
			row_flag[row[pos]][threadIdx.x] ^= flag;
			col_flag[col[pos]][threadIdx.x] ^= flag;
			box_flag[box[pos]][threadIdx.x] ^= flag;
			continue;
		}
		number[pos][threadIdx.x] += next_number[(row_flag[row[pos]][threadIdx.x] | col_flag[col[pos]][threadIdx.x] | box_flag[box[pos]][threadIdx.x]) >> (number[pos][threadIdx.x] + 1)] + 1;
		if (number[pos][threadIdx.x] >= ROW_NUM + 1) {
			number[pos][threadIdx.x] = 0;
			index--;
			if (index < 0) {
				continue;
			}
			pos = empty[index][threadIdx.x];
			flag = 1 << number[pos][threadIdx.x];
		} else {
			flag = 1 << number[pos][threadIdx.x];
			index++;
		}
		row_flag[row[pos]][threadIdx.x] ^= flag;
		col_flag[col[pos]][threadIdx.x] ^= flag;
		box_flag[box[pos]][threadIdx.x] ^= flag;
	}
	__syncthreads();
	if (threadIdx.x == 0 && found > 0) {
		atomicAdd(out_count, found);
	}
}
