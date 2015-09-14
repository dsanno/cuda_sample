#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_PROBLEM_NUM 1024
#define CELL_NUM 81
#define ROW_NUM 9

static int load(char *in_file_path, int *out_number);
static int solve(int *in_static_number, int in_max_count, int *out_result);

static const int row[] = {
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
static const int col[] = {
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
static const int box[] = {
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

int
main(int argc, char** argv) {
	int static_number[MAX_PROBLEM_NUM * CELL_NUM];
	int result[MAX_PROBLEM_NUM * CELL_NUM];
	clock_t start, stop;
	int problem_num;
	int answer_num[MAX_PROBLEM_NUM];
	int i, j, k;

	if (argc < 2 || argc >= 4) {
		printf("Usage: sudoku_cpu file_path [max_answer_num]");
		return 1;
	}
	char *file_path = argv[1];
	int max_count = 0x7fffffff;
	if (argc == 3) {
		max_count = strtol(argv[2], NULL, 10);
	}
	problem_num = load(file_path, static_number);
	if (problem_num <= 0) {
		printf("Can't load file %s.", file_path);
		return 1;
	}

	start = clock();
	for (i = 0; i < problem_num; i++) {
		answer_num[i] = solve(&static_number[i * CELL_NUM], max_count, &result[i * CELL_NUM]);
	}
	stop = clock();

	printf("%.6f (sec)\n", (double)(stop - start) / CLOCKS_PER_SEC);
	for (i = 0; i < problem_num; i++) {
		printf("anser count: %d\n", answer_num[i]);
		for (j = 0; j < ROW_NUM; j++) {
			for (k = 0; k < ROW_NUM; k++) {
				printf("%d ", result[j * ROW_NUM + k]);
			}
			printf("\n");
		}
	}
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
		} else if (buf[i] == '-') {
			out_number[n] = 0;
			n++;
		}
	}
	return n / CELL_NUM;
}

static int
solve(int *in_static_number, int in_max_count, int *out_result)
{
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

	index = 0;
	for (pos = 0; pos < CELL_NUM; pos++) {
		number[pos] = in_static_number[pos];
		if (in_static_number[pos] > 0) {
			flag = 1 << in_static_number[pos];
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
		}
		else {
			empty[index] = pos;
			index++;
		}
	}
	empty_num = index;
	index = 0;
	while (1) {
		pos = empty[index];
		for (number[pos]++; number[pos] < ROW_NUM + 1; number[pos]++) {
			flag = 1 << number[pos];
			if ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0) {
				continue;
			}
			if (index >= empty_num - 1) {
				for (i = 0; i < CELL_NUM; i++) {
					result[i] = number[i];
				}
				found++;
				result += CELL_NUM;
				if (found >= in_max_count) {
					break;
				}
			}
			else {
				break;
			}
		}
		if (found >= in_max_count) {
			break;
		}
		if (number[pos] < ROW_NUM + 1) {
			flag = 1 << number[pos];
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
			index++;
		}
		else {
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
