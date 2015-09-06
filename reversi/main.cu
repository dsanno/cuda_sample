#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/* types */

/* constant values */
#define ROW_STATE_NUM 6561
#define CELL_STATE_NUM 3
#define CELL_NUM 8
#define PATTERN_NUM (CELL_NUM * 8)
#define BLACK 0
#define WHITE 2
#define EMPTY 1
static const int pow3[] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561 };

#define BLOCK_SIZE 32
#define WARP_SIZE 32

/* macros */
#define OP_COLOR(c) (BLACK + WHITE - (c))

/* host proto types */
static void fill_flip_count(char out_flip_count[ROW_STATE_NUM][CELL_NUM]);
static void fill_reverse_pos(short int *out_reverse_pos);
static void fill_pattern_offset(short int out_pattern_offset[CELL_NUM * CELL_NUM][PATTERN_NUM]);
static int count_flip(const int *in_cells, int in_pos);

/* device constant values */
__constant__ char flip_count[ROW_STATE_NUM][CELL_NUM];
__constant__ short int reverse_pos[CELL_NUM];
__constant__ short int pattern_offset[CELL_NUM * CELL_NUM][PATTERN_NUM];

/* device proto types */
__global__ void flip_test(short int(*out_state)[PATTERN_NUM]);
__device__ void fill_reverse_state();
__device__ void init_state(int* in_board);
__device__ int flip(int in_n, int in_pos);


int main(int argc, char** argv){
	char host_flip_count[ROW_STATE_NUM][CELL_NUM];
	short int host_reverse_pos[CELL_NUM];
	short int host_pattern_offset[CELL_NUM * CELL_NUM][PATTERN_NUM];

	fill_flip_count(host_flip_count);
	fill_reverse_pos(host_reverse_pos);
	fill_pattern_offset(host_pattern_offset);
	cudaMemcpyToSymbol(flip_count, host_flip_count, sizeof(host_flip_count));
	cudaMemcpyToSymbol(reverse_pos, host_reverse_pos, sizeof(host_reverse_pos));
	cudaMemcpyToSymbol(pattern_offset, host_pattern_offset, sizeof(host_pattern_offset));

	/* ブロックサイズとグリッドサイズの設定 */
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(1, 1);

	cudaError_t error;
	cudaEvent_t start, stop;

	error = cudaEventCreate(&start);
	if (error != cudaSuccess) {
		printf("failed to craete start event");
		exit(EXIT_FAILURE);
	}
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

	short int host_state[32][PATTERN_NUM];
	short int (*state)[PATTERN_NUM];
	cudaMalloc((void**)&state, sizeof(host_state));
	cudaMemset(state, 0, sizeof(host_state));
	/* カーネルの起動 */
	flip_test<< <grid, block >> >(state);
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

	/* タイマーを停止 */
	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	if (error != cudaSuccess) {
		printf("failed to get elapsed time");
		exit(EXIT_FAILURE);
	}
	printf("Processing time: %f (msec)\n", msecTotal);

	cudaMemcpy(host_state, state, sizeof(host_state), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < PATTERN_NUM; j++) {
			printf("%d %d:", j, host_state[i][j]);
			int pattern = host_state[i][j];
			for (int k = 0; k < CELL_NUM; k++) {
				printf(" %c", "O.X"[pattern % 3]);
				pattern /= 3;
			}
			printf("\n");
		}
	}
	/* 終了処理 */
	cudaThreadExit();
}

static void
fill_flip_count(char out_flip_count[ROW_STATE_NUM][CELL_NUM])
{
	int cells[CELL_NUM];
	int i, j, n;

	for (i = 0; i < ROW_STATE_NUM; i++) {
		n = i;
		for (j = 0; j < CELL_NUM; j++) {
			cells[j] = n % 3;
			n /= 3;
		}
		for (j = 0; j < CELL_NUM; j++) {
			out_flip_count[i][j] = count_flip(cells, j);
		}
	}
}

static void
fill_reverse_pos(short int *out_reverse_pos)
{
	short int *t = out_reverse_pos;
	short int i;

	for (i = 0; i < CELL_NUM; i++) {
		*t = CELL_NUM - i;
		t++;
	}
}

static void
fill_pattern_offset(short int out_pattern_offset[CELL_NUM * CELL_NUM][PATTERN_NUM])
{
	int i, j;
	int base = 0;

	for (i = 0; i < CELL_NUM * CELL_NUM; i++) {
		for (j = 0; j < PATTERN_NUM; j++) {
			out_pattern_offset[i][j] = 0;
		}
	}
	/* horizontal */
	for (i = 0; i < CELL_NUM; i++) {
		for (j = 0; j < CELL_NUM; j++) {
			out_pattern_offset[i * CELL_NUM + j][i] = pow3[j];
		}
	}
	base += CELL_NUM;
	/* vertical */
	for (i = 0; i < CELL_NUM; i++) {
		for (j = 0; j < CELL_NUM; j++) {
			out_pattern_offset[j * CELL_NUM + i][i + base] = pow3[j];
		}
	}
	base += CELL_NUM;
	/* diagonal up right*/
	/* (0, i) to up right */
	for (i = 0; i < CELL_NUM; i++) {
		for (j = 0; j < i + 1; j++) {
			out_pattern_offset[(i - j) * CELL_NUM + j][i + base] = pow3[j];
		}
	}
	base += CELL_NUM;
	/* (i + 1, CELL_NUM - 1) to up right */
	for (i = 0; i < CELL_NUM; i++) {
		for (j = 0; j < CELL_NUM - i - 2; j++) {
			out_pattern_offset[(CELL_NUM - j - 1) * CELL_NUM + i + j + 1][i + base] = pow3[j];
		}
	}
	base += CELL_NUM;
	/* diagonal down right */
	/* (0, CELL_NUM - i - 1) to down right */
	for (i = 0; i < CELL_NUM; i++) {
		for (j = 0; j < i + 1; j++) {
			out_pattern_offset[(CELL_NUM - i + j - 1) * CELL_NUM + j][i + base] = pow3[j];
		}
	}
	base += CELL_NUM;
	/* (i + 1, 0) to down right */
	for (i = 0; i < CELL_NUM; i++) {
		for (j = 0; j < CELL_NUM - i - 2; j++) {
			out_pattern_offset[j * CELL_NUM + i + j + 1][i + base] = pow3[j];
		}
	}
	base += CELL_NUM;
}

static int
count_flip(const int *in_cell, int in_pos)
{
	int p;

	if (in_cell[in_pos] != EMPTY) {
		return 0;
	}
	for (p = in_pos + 1; p < CELL_NUM; p++) {
		if (in_cell[p] == EMPTY) {
			return 0;
		} else if (in_cell[p] == BLACK) {
			return p - in_pos - 1;
		}
	}
	return 0;
}

#define POS(x, y) ((y) * CELL_NUM + (x))

__shared__ short int state[30][PATTERN_NUM];
__shared__ short int reverse_state[ROW_STATE_NUM];

__global__ void
flip_test(short int(*out_state)[PATTERN_NUM])
{

	fill_reverse_state();

	int i, j;
	int board[CELL_NUM * CELL_NUM];
	for (i = 0; i < CELL_NUM * CELL_NUM; i++) {
		board[i] = EMPTY;
	}
	board[POS(3, 3)] = WHITE;
	board[POS(4, 3)] = BLACK;
	board[POS(3, 4)] = BLACK;
	board[POS(4, 4)] = WHITE;

	init_state(board);

	for (int n = 0; n < 1024 * 10; n++) {
		flip(0, POS(5, 4));
		flip(1, POS(3, 5));
		flip(2, POS(2, 2));
		flip(3, POS(3, 2));
		flip(4, POS(2, 3));
		flip(5, POS(5, 3));
		flip(6, POS(2, 4));
		flip(7, POS(1, 2));
		flip(8, POS(2, 1));
		flip(9, POS(4, 5));
	}

	if (threadIdx.x == 0) {
		for (i = 0; i < 30; i++) {
			for (j = 0; j < PATTERN_NUM; j++) {
				out_state[i][j] = state[i][j];
			}
		}
	}
}

__device__ void
fill_reverse_state()
{
	short int i, j, n, m;

	if (threadIdx.x != 0) {
		return;
	}
	for (i = 0; i < ROW_STATE_NUM; i++) {
		n = i;
		m = 0;
		for (j = 0; j < CELL_NUM; j++) {
			m *= 3;
			m += n % 3;
			n /= 3;
		}
		reverse_state[i] = m;
	}
}

__device__ void
init_state(int* in_board)
{
	int i, j;
	int base = 0;
	int pow3[] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561 };

	if (threadIdx.x != 0) {
		return;
	}

	/* horizontal */
	for (i = 0; i < CELL_NUM; i++) {
		state[0][i + base] = 0;
		for (j = 0; j < CELL_NUM; j++) {
			state[0][i + base] += in_board[i * CELL_NUM + j] * pow3[j];
		}
	}
	base += CELL_NUM;
	/* vertical */
	for (i = 0; i < CELL_NUM; i++) {
		state[0][i + base] = 0;
		for (j = 0; j < CELL_NUM; j++) {
			state[0][i + base] += in_board[j * CELL_NUM + i] * pow3[j];
		}
	}
	base += CELL_NUM;
	/* diagonal up right*/
	/* (0, i) to up right */
	for (i = 0; i < CELL_NUM; i++) {
		state[0][i + base] = 0;
		for (j = 0; j < i + 1; j++) {
			state[0][i + base] += in_board[(i - j) * CELL_NUM + j] * pow3[j];
		}
		for (; j < CELL_NUM; j++) {
			state[0][i + base] += EMPTY * pow3[j];
		}
	}
	base += CELL_NUM;
	/* (i + 1, CELL_NUM - 1) to up right */
	for (i = 0; i < CELL_NUM; i++) {
		state[0][i + base] = 0;
		for (j = 0; j < CELL_NUM - i - 2; j++) {
			state[0][i + base] += in_board[(CELL_NUM - j - 1) * CELL_NUM + i + j + 1] * pow3[j];
		}
		for (; j < CELL_NUM; j++) {
			state[0][i + base] += EMPTY * pow3[j];
		}
	}
	base += CELL_NUM;
	/* diagonal down right */
	/* (0, CELL_NUM - i - 1) to down right */
	for (i = 0; i < CELL_NUM; i++) {
		state[0][i + base] = 0;
		for (j = 0; j < i + 1; j++) {
			state[0][i + base] += in_board[(CELL_NUM - i + j - 1) * CELL_NUM + j] * pow3[j];
		}
		for (; j < CELL_NUM; j++) {
			state[0][i + base] += EMPTY * pow3[j];
		}
	}
	base += CELL_NUM;
	/* (i + 1, 0) to down right */
	for (i = 0; i < CELL_NUM; i++) {
		state[0][i + base] = 0;
		for (j = 0; j < CELL_NUM - i - 2; j++) {
			state[0][i + base] += in_board[j * CELL_NUM + i + j + 1] * pow3[j];
		}
		for (; j < CELL_NUM; j++) {
			state[0][i + base] += EMPTY * pow3[j];
		}
	}
	base += CELL_NUM;
	for (i = base; i < PATTERN_NUM; i++) {
		state[0][i] = 0;
		for (j = 0; j < CELL_NUM; j++) {
			state[0][i] += EMPTY * pow3[j];
		}
	}
}

__device__ int
flip(int in_n, int in_pos)
{
	int id = threadIdx.x;
	int x = in_pos % CELL_NUM;
	int y = in_pos / CELL_NUM;
	int rx = CELL_NUM - x - 1;
	int ry = CELL_NUM - y - 1;
	int count = 0;
	int base = 0;
	int offset;
	short int* current = state[in_n];
	short int* next = state[in_n + 1];
	int i;
	int result = 0;

	next[id] = ROW_STATE_NUM - current[id] - 1;
	next[id + WARP_SIZE] = ROW_STATE_NUM - current[id + WARP_SIZE] - 1;

	/* right */
	count = flip_count[current[y]][x];
	result += count;
	for (i = 0; i < count; i++) {
		next[id] += pattern_offset[in_pos + i + 1][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos + i + 1][id + WARP_SIZE] * 2;
	}

	/* left */
	count = flip_count[reverse_state[current[y]]][rx];
	result += count;
	for (i = 0; i < count; i++) {
		next[id] += pattern_offset[in_pos - i - 1][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos - i - 1][id + WARP_SIZE] * 2;
	}

	/* down */
	count = flip_count[current[x + CELL_NUM]][y];
	result += count;
	offset = 0;
	for (i = 0; i < count; i++) {
		offset += CELL_NUM;
		next[id] += pattern_offset[in_pos + offset][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos + offset][id + WARP_SIZE] * 2;
	}

	/* up */
	count = flip_count[reverse_state[current[x + CELL_NUM]]][ry];
	result += count;
	offset = 0;
	for (i = 0; i < count; i++) {
		offset += CELL_NUM;
		next[id] += pattern_offset[in_pos - offset][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos - offset][id + WARP_SIZE] * 2;
	}

	/* up right */
	if (x + y < CELL_NUM) {
		count = flip_count[current[x + y + CELL_NUM * 2]][x];
	} else {
		count = flip_count[current[x + y + CELL_NUM * 2]][ry];
	}
	result += count;
	offset = 0;
	for (i = 0; i < count; i++) {
		offset += CELL_NUM - 1;
		next[id] += pattern_offset[in_pos - offset][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos - offset][id + WARP_SIZE] * 2;
	}

	/* down left */
	if (x + y < CELL_NUM) {
		count = flip_count[reverse_state[current[x + y + CELL_NUM * 2]]][rx];
	}
	else {
		count = flip_count[reverse_state[current[x + y + CELL_NUM * 2]]][y];
	}
	result += count;
	offset = 0;
	for (i = 0; i < count; i++) {
		offset += CELL_NUM - 1;
		next[id] += pattern_offset[in_pos + offset][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos + offset][id + WARP_SIZE] * 2;
	}

	/* down right */
	if (x + ry < CELL_NUM) {
		count = flip_count[current[x + ry + CELL_NUM * 4]][x];
	}
	else {
		count = flip_count[current[x + ry + CELL_NUM * 4]][y];
	}
	result += count;
	offset = 0;
	for (i = 0; i < count; i++) {
		offset += CELL_NUM + 1;
		next[id] += pattern_offset[in_pos + offset][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos + offset][id + WARP_SIZE] * 2;
	}

	/* up left */
	if (x + ry < CELL_NUM) {
		count = flip_count[reverse_state[current[x + ry + CELL_NUM * 4]]][rx];
	}
	else {
		count = flip_count[reverse_state[current[x + ry + CELL_NUM * 4]]][ry];
	}
	result += count;
	offset = 0;
	for (i = 0; i < count; i++) {
		offset += CELL_NUM + 1;
		next[id] += pattern_offset[in_pos - offset][id] * 2;
		next[id + WARP_SIZE] += pattern_offset[in_pos - offset][id + WARP_SIZE] * 2;
	}


	if (result > 0) {
		next[id] += pattern_offset[in_pos][id];
		next[id + WARP_SIZE] += pattern_offset[in_pos][id + WARP_SIZE];
	}
	return result;
}
