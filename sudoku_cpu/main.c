#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
	int static_number[81];
	int result[81];
	clock_t start, stop;
	int answer_num;

	if (argc < 2 || argc >= 4) {
		printf("Usage: sudoku_cpu file_path [max_answer_num]");
		return 1;
	}
	char *file_path = argv[1];
	int max_count = 0x7fffffff;
	if (argc == 3) {
		max_count = strtol(argv[2], NULL, 10);
	}
	if (!load(file_path, static_number)) {
		printf("Can't load file %s.", file_path);
		return 1;
	}

	start = clock();
	answer_num = solve(static_number, max_count, result);
	stop = clock();

	printf("%.6f (sec)\n", (double)(stop - start) / CLOCKS_PER_SEC);
	printf("anser count: %d\n", answer_num);
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			printf("%d ", result[i * 9 + j]);
		}
		printf("\n");
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

	for (i = 0, n = 0; i < size && n < 81; i++) {
		if (buf[i] >= '1' && buf[i] <= '9') {
			out_number[n] = buf[i] - '0';
			n++;
		} else if (buf[i] == '-') {
			out_number[n] = 0;
			n++;
		}
	}
	if (n < 81) {
		return 0;
	}
	return 1;
}

static int
solve(int *in_static_number, int in_max_count, int *out_result) {
	int row_flag[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int col_flag[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int box_flag[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int number[81];
	int found = 0;
	int pos;
	int flag;
#if 1
	int count = 0;
	char nn[16384][50];
#endif

	for (pos = 0; pos < 81; pos++) {
		number[pos] = in_static_number[pos];
		if (in_static_number[pos] > 0) {
			flag = 1 << in_static_number[pos];
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
		}
	}
	pos = 0;
	while (pos < 81 && in_static_number[pos] > 0) {
		pos++;
	}
	while (1) {
		if (pos < 0) {
			break;
		}
#if 1 // test
		if (pos >= 50) {
			if (found < 16384) {
				for (int i = 0; i < 50; i++) {
					nn[found][i] = number[i];
				}
			}
			found++;
			pos--;
			while (pos >= 0 && in_static_number[pos] > 0) {
				pos--;
			}
			if (pos >= 0) {
				flag = 1 << number[pos];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
			}
			continue;
		}
#endif
		if (pos >= 81) {
			if (found == 0) {
				for (int i = 0; i < 81; i++) {
					out_result[i] = number[i];
				}
			}
			found++;
			if (found >= in_max_count) {
				break;
			}
			pos--;
			while (pos >= 0 && in_static_number[pos] > 0) {
				pos--;
			}
			if (pos >= 0) {
				flag = 1 << number[pos];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
			}
			continue;
		}
		do {
			number[pos]++;
			flag = 1 << number[pos];
		} while (number[pos] <= 9 && ((row_flag[row[pos]] & flag) != 0 || (col_flag[col[pos]] & flag) != 0 || (box_flag[box[pos]] & flag) != 0));
		if (number[pos] > 9) {
			number[pos] = 0;
			pos--;
			while (pos >= 0 && in_static_number[pos] > 0) {
				pos--;
			}
			if (pos >= 0) {
				flag = 1 << number[pos];
				row_flag[row[pos]] &= ~flag;
				col_flag[col[pos]] &= ~flag;
				box_flag[box[pos]] &= ~flag;
			}
		} else {
			row_flag[row[pos]] |= flag;
			col_flag[col[pos]] |= flag;
			box_flag[box[pos]] |= flag;
			pos++;
			while (pos < 81 && in_static_number[pos] > 0) {
				pos++;
			}
		}
	}
	return found;
}
