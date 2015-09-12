#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_GRID_SIZE 65536
#define GRID_SIZE 1
#define BLOCK_SIZE 729
#define CANDIDATE_NUM 27
#define CANDIDATE_WIDTH 729

static int find_valid_number(const int *in_static_number, int in_depth, int in_max_count, int *out_result);

__global__ void solve_sudoku(int *in_static_number, int *out_result, int *out_count);
__constant__ int candidate[CANDIDATE_WIDTH][3];

const int host_static_number[] = {
	0, 0, 5, 3, 0, 0, 0, 0, 0,
	8, 0, 0, 0, 0, 0, 0, 2, 0,
	0, 7, 0, 0, 1, 0, 5, 0, 0,
	4, 0, 0, 0, 0, 5, 3, 0, 0,
	0, 1, 0, 0, 7, 0, 0, 0, 6,
	0, 0, 3, 2, 0, 0, 0, 8, 0,
	0, 6, 0, 5, 0, 0, 0, 0, 9,
	0, 0, 4, 0, 0, 0, 0, 3, 0,
	0, 0, 0, 0, 0, 9, 7, 0, 0
};

int main(int argc, char** argv){
	int *device_result;
	int *device_count;
	int *device_static_number;
	int count;
	int result[81];
	int host_candidate[CANDIDATE_WIDTH][3];
	int *valid_number;
	int i, j;

	for (i = 0; i < CANDIDATE_WIDTH; i++) {
		host_candidate[i][0] = i / (9 * 9) + 1;
		host_candidate[i][1] = i / 9 % 9 + 1;
		host_candidate[i][2] = i % 9 + 1;
	}
	cudaMalloc((void**)&device_result, sizeof(int) * 81);
	cudaMemset(device_result, 0, sizeof(int) * 81);
	cudaMalloc((void**)&device_count, sizeof(int));
	cudaMemset(device_count, 0, sizeof(int));
	cudaMemcpyToSymbol(candidate, host_candidate, sizeof(candidate));
	valid_number = (int*)malloc(sizeof(int) * 81 * MAX_GRID_SIZE);
	cudaMalloc((void**)&device_static_number, sizeof(int) * 81 * MAX_GRID_SIZE);

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

	count = find_valid_number(host_static_number, 14, MAX_GRID_SIZE, valid_number);
	cudaMemcpy(device_static_number, valid_number, sizeof(int) * 81 * count, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(count, 1);
	solve_sudoku << <grid, block >> >(device_static_number, device_result, device_count);
	cudaThreadSynchronize();

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

	cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d found\n", count);
	cudaMemcpy(result, device_result, sizeof(int) * 81, cudaMemcpyDeviceToHost);
	for (i = 0; i < 9; i++) {
		for (j = 0; j < 9; j++) {
			printf("%d ", result[i * 9 + j]);
		}
		printf("\n");
	}
	cudaFree(device_result);
	cudaFree(device_count);
	free(valid_number);

	cudaThreadExit();
}

static int
find_valid_number(const int *in_static_number, int in_depth, int in_max_count, int* out_result) {
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
	int row_flag[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int col_flag[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int box_flag[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int empty[81];
	int number[81];
	int found = 0;
	int pos;
	int flag;
	int index;
	int empty_num;
	int i;
	int *result = out_result;

	index = 0;
	for (pos = 0; pos < 81; pos++) {
		number[pos] = in_static_number[pos];
		if (in_static_number[pos] > 0) {
			flag = 1 << in_static_number[pos];
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
		} else {
			empty[index] = pos;
			index++;
		}
	}
	empty_num = index;
	index = 0;
	while (1) {
		pos = empty[index];
		for (number[pos]++; number[pos] < 10; number[pos]++) {
			flag = 1 << number[pos];
			if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
				continue;
			}
			if (index >= in_depth - 1 || index >= empty_num - 1) {
				for (i = 0; i < 81; i++) {
					result[i] = number[i];
				}
				found++;
				result += 81;
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
		if (number[pos] < 10) {
			flag = 1 << number[pos];
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
			index++;
		} else {
			index--;
			if (index < 0) {
				break;
			}
			number[pos] = 0;
			pos = empty[index];
			flag = 1 << number[pos];
			row_flag[row[pos]] &= ~flag;
			col_flag[col[pos]] &= ~flag;
			box_flag[box[pos]] &= ~flag;
		}
	}
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
	__shared__ int row_flag[9];
	__shared__ int col_flag[9];
	__shared__ int box_flag[9];
	__shared__ int number[81];
	__shared__ int empty[81];
	__shared__ short int valid_candidate[CANDIDATE_NUM][CANDIDATE_WIDTH];
	__shared__ int candidate_index[CANDIDATE_NUM];
	__shared__ int candidate_count[CANDIDATE_NUM];
	__shared__ int empty_num;
	int pos;
	int index;
	int flag;
	int r[3];
	int empty_index;
	int is_valid;
	int left;
	int i;

	if (threadIdx.x == 0) {
		found = 0;
		for (index = 0; index < 9; index++) {
			row_flag[index] = 0;
			col_flag[index] = 0;
			box_flag[index] = 0;
		}
		index = 0;
		for (pos = 0; pos < 81; pos++) {
			number[pos] = in_static_number[blockIdx.x * 81 + pos];
			if (number[pos] > 0) {
				flag = 1 << number[pos];
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
			} else {
				empty[index] = pos;
				index++;
			}
		}
		empty_num = index;
		for (index = 0; index < CANDIDATE_NUM; index++) {
			candidate_index[index] = 0;
			candidate_count[index] = 0;
		}
	}

	__syncthreads();
	index = 0;
	empty_index = 0;
	while (found < 0x7fffffff) {
		left = empty_num - empty_index;
		if (candidate_index[index] == 0) {
			if (left >= 3) {
				is_valid = 1;
				pos = empty[empty_index];
				flag = 1 << candidate[threadIdx.x][0];
				r[0] = row[pos];
				if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
					is_valid = 0;
				}
				if (is_valid != 0) {
					pos = empty[empty_index + 1];
					flag = 1 << candidate[threadIdx.x][1];
					r[1] = row[pos];
					if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
						is_valid = 0;
					}
				}
				if (is_valid != 0) {
					pos = empty[empty_index + 2];
					flag = 1 << candidate[threadIdx.x][2];
					r[2] = row[pos];
					if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
						is_valid = 0;
					}
				}
				if (is_valid != 0) {
					if (r[0] == r[1] && candidate[threadIdx.x][0] == candidate[threadIdx.x][1]) {
						is_valid = 0;
					} else if (r[1] == r[2] && candidate[threadIdx.x][1] == candidate[threadIdx.x][2]) {
						is_valid = 0;
					} else if (r[0] == r[2] && candidate[threadIdx.x][0] == candidate[threadIdx.x][2]) {
						is_valid = 0;
					}
				}
				if (is_valid != 0) {
					valid_candidate[index][atomicAdd(&candidate_count[index], 1)] = threadIdx.x;
					if (left == 3 && atomicAdd(&found, 1) == 0) {
						for (i = 0; i < 78; i++) {
							out_result[i] = number[i];
						}
						out_result[empty[empty_index]] = candidate[threadIdx.x][0];
						out_result[empty[empty_index + 1]] = candidate[threadIdx.x][1];
						out_result[empty[empty_index + 2]] = candidate[threadIdx.x][2];
					}
				}
			} else if (left == 2) {
				if (threadIdx.x < 81) {
					is_valid = 1;
					pos = empty[empty_index];
					flag = 1 << candidate[threadIdx.x][1];
					r[0] = row[pos];
					if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
						is_valid = 0;
					}
					pos = empty[empty_index + 1];
					flag = 1 << candidate[threadIdx.x][2];
					r[1] = row[pos];
					if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
						is_valid = 0;
					}
					if (r[0] == r[1] && candidate[threadIdx.x][1] == candidate[threadIdx.x][2]) {
						is_valid = 0;
					}
					if (is_valid) {
						if (atomicAdd(&found, 1) == 0) {
							for (i = 0; i < 79; i++) {
								out_result[i] = number[i];
							}
							out_result[empty[empty_index]] = candidate[threadIdx.x][1];
							out_result[empty[empty_index + 1]] = candidate[threadIdx.x][2];
						}
					}
				}
			} else if (left == 1) {
				if (threadIdx.x < 9) {
					is_valid = 1;
					pos = empty[empty_index];
					flag = 1 << candidate[threadIdx.x][2];
					if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
						is_valid = 0;
					}
					if (is_valid) {
						if (atomicAdd(&found, 1) == 0) {
							for (i = 0; i < 80; i++) {
								out_result[i] = number[i];
							}
							out_result[empty[empty_index]] = candidate[threadIdx.x][2];
						}
					}

				}
			}
		}
		__syncthreads();
		if (left > 3 && candidate_index[index] < candidate_count[index]) {
			if (threadIdx.x == 0) {
				i = valid_candidate[index][candidate_index[index]];
				pos = empty[empty_index];
				number[pos] = candidate[i][0];
				flag = 1 << candidate[i][0];
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
				pos = empty[empty_index + 1];
				number[pos] = candidate[i][1];
				flag = 1 << candidate[i][1];
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
				pos = empty[empty_index + 2];
				number[pos] = candidate[i][2];
				flag = 1 << candidate[i][2];
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
				candidate_index[index]++;
			}
			index++;
			empty_index += 3;
		} else {
			index--;
			empty_index -= 3;
			if (index < 0) {
				break;
			}
			if (threadIdx.x == 0) {
				i = valid_candidate[index][candidate_index[index] - 1];
				candidate_index[index + 1] = 0;
				candidate_count[index + 1] = 0;
				pos = empty[empty_index];
				flag = 1 << candidate[i][0];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
				pos = empty[empty_index + 1];
				flag = 1 << candidate[i][1];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
				pos = empty[empty_index + 2];
				flag = 1 << candidate[i][2];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
			}
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x == 0 && found > 0) {
		atomicAdd(out_count, found);
	}
}
