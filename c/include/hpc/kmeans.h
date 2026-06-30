#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <hpc/types.h>
#include <hpc/dataset.h>

struct kmeans {
	f64* centroids;
	u32  n_clusters;
	u32  max_iter;
};

struct kmeans kmeans_init(u32 n_clusters, u32 max_iter);
void	      kmeans_deinit(struct kmeans* km);

void kmeans_fit(struct kmeans* km, struct dataset* X);
void kmeans_predict(struct kmeans* km, struct dataset* y, u32* result);
void kmeans_fit_predict(struct kmeans* km, struct dataset* X, struct dataset* y,
			u32* result);

#ifdef MPI_ENABLED
struct mpi_ctx {
	i32 rank;
	i32 num_procs;
};

void kmeans_mpi_fit(struct mpi_ctx* ctx, struct kmeans* km, struct dataset* X);
#endif // !MPI_ENABLED

#endif // !__KMEANS_H__
