#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define NEW_SOLVER

#ifdef NEW_SOLVER
#define GRID_SIZE 1
#define BLOCK_SIZE 504
#define CANDIDATE_NUM 27
#define CANDIDATE_WIDTH 504
#else
#define GRID_SIZE 1024
#define BLOCK_SIZE 32
#endif

__global__ void solve_sudoku(int *out_result, int *out_count);
__constant__ int candidate[CANDIDATE_WIDTH][3];

int main(int argc, char** argv){
	int *device_result;
	int *device_count;
	int count;
	int result[81];
	int host_candidate[CANDIDATE_WIDTH][3];

	for (int i = 0; i < CANDIDATE_WIDTH; i++) {
		host_candidate[i][0] = i / (7 * 8) + 1;
		host_candidate[i][0] = i / 7 % 8 + 1;
		host_candidate[i][2] = i % 7 + 1;
	}
	cudaMalloc((void**)&device_result, sizeof(int) * 81);
	cudaMemset(device_result, 0, sizeof(int) * 81);
	cudaMalloc((void**)&device_count, sizeof(int));
	cudaMemset(device_count, 0, sizeof(int));
	cudaMemcpyToSymbol(candidate, host_candidate, sizeof(candidate));

	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(GRID_SIZE, 1);

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

	solve_sudoku << <grid, block >> >(device_result, device_count);
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
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			printf("%d ", result[i * 9 + j]);
		}
		printf("\n");
	}
	cudaFree(device_result);
	cudaFree(device_count);

	cudaThreadExit();
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
__constant__ int static_number[] = {
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

#ifdef NEW_SOLVER
__global__ void
solve_sudoku(int *out_result, int *out_count) {
	__shared__ int found;
	__shared__ char valid[CANDIDATE_NUM][CANDIDATE_WIDTH];
	__shared__ int row_flag[9];
	__shared__ int col_flag[9];
	__shared__ int box_flag[9];
	__shared__ int number[81];
	__shared__ int empty[81];
	__shared__ int candidate_index[CANDIDATE_NUM];
	__shared__ int empty_num;
	int pos;
	int index;
	int flag;
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
			number[pos] = static_number[pos];
			if (static_number[pos] > 0) {
				flag = 1 << static_number[pos];
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
			candidate_index[index] = -1;
		}
	}

	__syncthreads();
	index = 0;
	empty_index = 0;
	while (1) {
		left = empty_num - empty_index;
		// for debug
		if (index > 11) {
			break;
		}
		if (threadIdx.x == 0) {
			atomicAdd(&found, 1);
		}
		if (candidate_index[11] > 0) {
			if (threadIdx.x == 0) {
				found = 777;
			}
			break;
		}
		if (found > 100000000) {
			break;
		}
		//
		if (candidate_index[index] < 0) {
			if (left >= 3) {
				is_valid = 1;
				pos = empty[empty_index];
				flag = 1 << candidate[threadIdx.x][0];
				if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
					is_valid = 0;
				}
				pos = empty[empty_index + 1];
				flag = 1 << candidate[threadIdx.x][1];
				if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
					is_valid = 0;
				}
				pos = empty[empty_index + 2];
				flag = 1 << candidate[threadIdx.x][2];
				if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
					is_valid = 0;
				}
				valid[index][threadIdx.x] = is_valid;
			} else if (left == 2) {
				int *x = NULL;
				*x = 0;
				atomicAdd(&found, 1);
				break;
			} else if (left == 1) {
				int *x = NULL;
				*x = 0;
				atomicAdd(&found, 1);
				break;
			}
			else {
				int *x = NULL;
				*x = 0;
				atomicAdd(&found, 1);
				break;
			}
		}
		__syncthreads();
		if (threadIdx.x == 0 && left >= 3) {
			for (candidate_index[index]++; candidate_index[index] < CANDIDATE_WIDTH; candidate_index[index]++) {
				if (valid[index][candidate_index[index]] != 0) {
					break;
				}
			}
		}
		__syncthreads();
		// for debug
		if (index > 10 && candidate_index[11] > 0 && candidate_index[11] < CANDIDATE_WIDTH) {
			if (threadIdx.x == 0) {
				found = 666;
			}
			break;
		}
		//
		if (left >= 3 && candidate_index[index] < CANDIDATE_WIDTH) {
			if (threadIdx.x == 0) {
				pos = empty[empty_index];
				flag = 1 << candidate[candidate_index[index]][0];
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
				pos = empty[empty_index + 1];
				flag = 1 << candidate[candidate_index[index]][1];
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
				pos = empty[empty_index + 2];
				flag = 1 << candidate[candidate_index[index]][2];
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
			}
			index++;
			empty_index += 3;
		} else {
			// for debug
			if (threadIdx.x == 0) {
				if (left < 3) {
					atomicAdd(&found, 1000000);
				}
				if (candidate_index[index] >= CANDIDATE_WIDTH) {
					atomicAdd(&found, 10000);
				}
			}
			// 
			index--;
			empty_index -= 3;
			if (index < 0) {
				break;
			}
			if (threadIdx.x == 0) {
				candidate_index[index + 1] = -1;
				pos = empty[empty_index];
				flag = 1 << candidate[candidate_index[index]][0];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
				pos = empty[empty_index + 1];
				flag = 1 << candidate[candidate_index[index]][1];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
				pos = empty[empty_index + 2];
				flag = 1 << candidate[candidate_index[index]][2];
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
#else
__global__ void
solve_sudoku(int *out_result, int *out_count) {
	__shared__ int found;
	int row_flag[9] = {};
	int col_flag[9] = {};
	int box_flag[9] = {};
#if 0
	int static_number[] = {
		1, 2, 3, 4, 5, 6, 7, 8, 9,
		4, 5, 6, 7, 8, 9, 1, 2, 3,
		7, 8, 9, 1, 2, 3, 4, 5, 6,
		2, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0
	};
#else
	int static_number[] = {
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
#endif
	int number[81] = {};
	int empty[81] = {};
	int pos;
	int index;
	int flag;

	index = 0;
	for (pos = 0; pos < 81; pos++) {
		number[pos] = static_number[pos];
		if (static_number[pos] > 0) {
			flag = 1 << static_number[pos];
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
		} else {
			empty[index] = pos;
			index++;
		}
	}
	empty[index] = -1;
	__shared__ int next_seed;
	int seed_offset = gridDim.x;
	int seed = blockIdx.x + seed_offset * threadIdx.x;
	int seed_left = seed;
	int seed_index = 6;
	int seed_max = 9 * 9 * 9 * 9 * 9 * 9;
	if (threadIdx.x == 0) {
		found = 0;
		next_seed = blockIdx.x + seed_offset * blockDim.x;
	}
	__syncthreads();
	index = 0;
	while (1) {
		if (index < 0) {
			seed = atomicAdd(&next_seed, seed_offset);
			seed_left = seed;
			if (seed >= seed_max) {
				break;
			} else {
				index = 0;
			}
		}
		pos = empty[index];
		if (pos < 0) {
			int old_found = atomicAdd(&found, 1);
			if (old_found == 0) {
				for (int i = 0; i < 81; i++) {
					out_result[i] = number[i];
				}
			}
			index--;
			pos = empty[index];
			flag = 1 << number[pos];
			row_flag[row[pos]] &= ~flag;
			col_flag[col[pos]] &= ~flag;
			box_flag[box[pos]] &= ~flag;
			continue;
		}
		if (index < seed_index) {
#if 1
			for (; index < seed_index; index++) {
				pos = empty[index];
				number[pos] = seed_left % 9 + 1;
				seed_left /= 9;
				flag = 1 << number[pos];
				if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
					break;
				}
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
			}
			if (index < seed_index) {
				for (index--; index >= 0; index--) {
					pos = empty[index];
					flag = 1 << number[pos];
					row_flag[row[pos]] &= ~flag;
					col_flag[col[pos]] &= ~flag;
					box_flag[box[pos]] &= ~flag;
				}
				continue;
			}
#else
			number[pos] = seed_left % 9 + 1;
			seed_left /= 9;
			flag = 1 << number[pos];
			if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
				for (index--; index >= 0; index--) {
					pos = empty[index];
					flag = 1 << number[pos];
					row_flag[row[pos]] &= ~flag;
					col_flag[col[pos]] &= ~flag;
					box_flag[box[pos]] &= ~flag;
				}
			} else {
				row_flag[row[pos]] |= flag;
				col_flag[col[pos]] |= flag;
				box_flag[box[pos]] |= flag;
				index++;
			}
#endif
			continue;
		}
		number[pos]++;
		flag = 1 << number[pos];
		if (number[pos] <= 9 && ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0)) {
			continue;
		}
		if (number[pos] > 9) {
			number[pos] = 0;
			index--;
			pos = empty[index];
			flag = 1 << number[pos];
			row_flag[row[pos]] &= ~flag;
			col_flag[col[pos]] &= ~flag;
			box_flag[box[pos]] &= ~flag;
			if (index < seed_index) {
				for (index--; index >= 0; index--) {
					pos = empty[index];
					flag = 1 << number[pos];
					row_flag[row[pos]] &= ~flag;
					col_flag[col[pos]] &= ~flag;
					box_flag[box[pos]] &= ~flag;
				}
			}
		} else {
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
			index++;
		}
	}
	__syncthreads();
	if (threadIdx.x == 0 && found > 0) {
		atomicAdd(out_count, found);
	}
}
#endif
