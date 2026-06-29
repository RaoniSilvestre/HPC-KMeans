#ifndef __DATASET_H__

#include <hpc/types.h>

struct dataset {
	f64*  data;
	usize len;
	u32   n_feats;
};

struct dataset from_file(char* path);
struct dataset from_cmdline(int argc, char** argv);

struct holdout {
	struct dataset X_train;
	struct dataset X_test;
};

struct holdout train_test_split(struct dataset* X, f64 split);

#endif // !__DATASET_H__
