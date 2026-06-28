#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hpc/dataset.h>

struct holdout train_test_split(struct dataset* X, f64 split)
{
	assert(split < 1 && split > 0);
	usize train_len = X->len * split;
	usize remaining_len = X->len - train_len;

	struct dataset X_train = {
		.data = X->data,
		.n_feats = X->n_feats,
		.len = train_len,
	};
	struct dataset X_test = {
		.data = &X->data[train_len * X->n_feats],
		.n_feats = X->n_feats,
		.len = remaining_len,
	};

	return (struct holdout){
		.X_train = X_train,
		.X_test = X_test,
	};
}

struct dataset read_from_file(char* path)
{
	FILE* fp = fopen(path, "r");
	if (!fp) {
		printf("file doesn't exists\n");
		exit(1);
	}

	f64*  data = malloc(sizeof(f64) * 256);
	usize len = 0, capacity = 256;
	usize width = 0;
	bool  counted = false;

	char* line = 0;
	usize line_len = 0;
	i32   read;
	while ((read = getline(&line, &line_len, fp)) != -1) {
		char* tok = strtok(line, "\t");
		while (tok) {
			if (!counted)
				width++;
			if (len == capacity) {
				capacity *= 2;
				data = realloc(data, capacity * sizeof(f64));
			}
			data[len++] = strtod(tok, NULL);
			tok = strtok(NULL, "\t");
		}
		counted = true;

		// Assuming that the last column is the label. We're removing it
		if (len > 1)
			len--;
	}
	if (width > 1)
		width--;

	free(line);

	return (struct dataset){
		.data = data,
		.len = len / width,
		.n_feats = width,
	};
}
