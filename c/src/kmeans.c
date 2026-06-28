#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include <hpc/algebra.h>
#include <hpc/kmeans.h>
#include <hpc/types.h>

#define IDX(i, dim) ((i) * (dim))

static inline u32* get_seed(void)
{
	static _Thread_local u32 seed = 0;
	if (seed == 0)
		seed = (u32)time(NULL) ^ ((u32)omp_get_thread_num() << 16);
	return &seed;
}

static inline usize rand_pos(usize len)
{
	return rand_r(get_seed()) % len;
}

static usize rand_pos_with_probability(f64* p, usize len)
{
	u32   seed = (u32)time(NULL);
	f64   r = (f64)rand_r(&seed) / RAND_MAX;
	usize chosen = 0;
	f64   sum = 0;
	for (usize i = 0; i < len; i++) {
		sum += p[i];
		if (r <= sum) {
			chosen = i;
			break;
		}
	}
	return chosen;
}

struct closest_pt {
	f64   dist;
	usize idx;
};

static struct closest_pt min_dist(f64* pt, f64* pts, usize pt_dim,
				  usize pts_len)
{
	f64   min = INFINITY;
	usize closest = -1;
	for (usize i = 0; i < pts_len; i++) {
		f64 dist = euclidean_dist(pt, &pts[IDX(i, pt_dim)], pt_dim);
		if (dist < min) {
			min = dist;
			closest = i;
		}
	}
	return (struct closest_pt){ .dist = min, .idx = closest };
}

struct kmeans kmeans_init(u32 n_clusters, u32 max_iter)
{
	assert(n_clusters > 0);
	assert(max_iter > 0);

	return (struct kmeans){
		.centroids = NULL,
		.labels = NULL,
		.n_clusters = n_clusters,
		.max_iter = max_iter,
	};
}

void kmeans_deinit(struct kmeans* km)
{
	free(km->centroids);
	free(km->labels);
}

static inline void init_centroids(struct kmeans* km, struct dataset* X)
{
	f64* centroids = malloc(sizeof(f64) * km->n_clusters * X->n_feats);
	assert(centroids != NULL);
	usize centroids_len = 0;

	f64* distances = malloc(sizeof(f64) * X->len);
	assert(distances != NULL);
	usize distances_last = 0;

	vec_assign(&centroids[IDX(centroids_len++, X->n_feats)],
		   &X->data[IDX(rand_pos(X->len), X->n_feats)], X->n_feats);

	for (usize i = 0; i < km->n_clusters - 1; i++) {
		for (usize j = 0; j < X->len; j++) {
			struct closest_pt min =
				min_dist(&X->data[IDX(j, X->n_feats)],
					 centroids, X->n_feats, centroids_len);
			distances[distances_last++] = min.dist;
		}
		vec_sq(distances, distances_last);
		f64 sum = vec_sum(distances, distances_last);
		vec_mul_scalar(distances, 1 / sum, distances_last);

		usize next_centroid =
			rand_pos_with_probability(distances, distances_last);

		vec_assign(&centroids[IDX(centroids_len++, X->n_feats)],
			   &X->data[IDX(next_centroid, X->n_feats)],
			   X->n_feats);

		memset(distances, 0, distances_last);
		distances_last = 0;
	}

	km->centroids = centroids;
}

static bool update_labels(u32* labels, usize n_feats, f64* centroids,
			  u32 n_clusters, f64* data, usize len)
{
	bool changed = false;
#pragma omp parallel for
	for (usize j = 0; j < len; j++) {
		struct closest_pt cluster = min_dist(
			&data[IDX(j, n_feats)], centroids, n_feats, n_clusters);
		if (labels[j] != cluster.idx) {
			labels[j] = cluster.idx;
			changed = true;
		}
	}
	return changed;
}

static void update_centroids(f64* centroids, u32* labels, usize n_feats,
			     u32 n_clusters, f64* data, usize len)
{
	f64* count = calloc(n_clusters, sizeof(f64));
	assert(count != NULL);
	f64* sums = calloc(n_clusters * n_feats, sizeof(f64));
	assert(sums != NULL);

#pragma omp parallel for reduction(+ : count[ : n_clusters], \
					   sums[ : n_clusters * n_feats])
	for (usize p = 0; p < len; p++) {
		count[labels[p]]++;

		for (usize f = 0; f < n_feats; f++) {
			sums[IDX(labels[p], n_feats) + f] +=
				data[IDX(p, n_feats) + f];
		}
	}
	for (usize c = 0; c < n_clusters; c++) {
		f64* centroid = &centroids[IDX(c, n_feats)];
		if (count[c] == 0) {
			vec_assign(&centroids[IDX(c, n_feats)],
				   &data[IDX(rand_pos(len), n_feats)], n_feats);
			continue;
		}
		for (usize f = 0; f < n_feats; f++) {
			centroid[f] = sums[IDX(c, n_feats) + f] / count[c];
		}
	}
}

void kmeans_fit(struct kmeans* km, struct dataset* X)
{
	assert(km != NULL);
	assert(X != NULL);

	init_centroids(km, X);

	u32* labels = calloc(X->len, sizeof(u32));
	assert(labels != NULL);

	for (usize i = 0; i < km->max_iter; i++) {
		bool changed = update_labels(labels, X->n_feats, km->centroids,
					     km->n_clusters, X->data, X->len);
		update_centroids(km->centroids, labels, X->n_feats,
				 km->n_clusters, X->data, X->len);

		if (!changed)
			break;
	}

	km->labels = labels;
}

void kmeans_predict(struct kmeans* km, struct dataset* y, u32* result)
{
	assert(y != NULL);
	assert(km != NULL);
	assert(result != NULL);

#pragma omp parallel for
	for (usize i = 0; i < y->len; i++) {
		struct closest_pt cluster =
			min_dist(&y->data[IDX(i, y->n_feats)], km->centroids,
				 y->n_feats, km->n_clusters);
		result[i] = cluster.idx;
	}
}

void kmeans_fit_predict(struct kmeans* km, struct dataset* X, struct dataset* y,
			u32* result)
{
	kmeans_fit(km, X);
	kmeans_predict(km, y, result);
}

#ifdef MPI_ENABLED
#include <mpi.h>
#include <stdio.h>

static inline void scatter_data(struct mpi_ctx* ctx, f64* km_centroids,
				usize n_clusters, f64* data, usize* len,
				f64** centroids, u32* n_feats, i32** sendcounts,
				i32** displs, f64** recvbuf, i32* recvcount)
{
	MPI_Bcast(len, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(n_feats, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

	*centroids = malloc(sizeof(f64) * *n_feats * n_clusters);
	if (!*centroids)
		goto alloc_error;

	if (ctx->rank == 0) {
		memcpy(*centroids, km_centroids,
		       sizeof(f64) * *n_feats * n_clusters);
	}

	MPI_Bcast(*centroids, *n_feats * n_clusters, MPI_DOUBLE, 0,
		  MPI_COMM_WORLD);

	*sendcounts = malloc(sizeof(i32) * ctx->num_procs);
	if (!*sendcounts)
		goto alloc_error;

	*displs = malloc(sizeof(i32) * ctx->num_procs);
	if (!*displs)
		goto alloc_error;

	i32 rem = *len;
	i32 start = 0;
	i32 slice = *len / ctx->num_procs;
	for (usize i = 0; i < (usize)ctx->num_procs; i++) {
		(*sendcounts)[i] = slice * *n_feats;
		(*displs)[i] = start * *n_feats;
		start += slice;
		rem -= slice;
	}
	(*sendcounts)[ctx->num_procs - 1] += rem * *n_feats;

	*recvcount = (*sendcounts)[ctx->rank];
	*recvbuf = malloc(sizeof(f64) * *recvcount);
	if (!*recvbuf)
		goto alloc_error;

	MPI_Scatterv(data, *sendcounts, *displs, MPI_DOUBLE, *recvbuf,
		     *recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	return;

alloc_error:
	fprintf(stderr, "allocation failure\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
}

static inline void mpi_update_centroids(f64* centroids, u32* labels,
					usize n_feats, u32 n_clusters,
					f64* data, usize len)
{
	f64* local_count = calloc(n_clusters, sizeof(f64));
	if (!local_count)
		goto alloc_error;
	f64* local_sum = calloc(n_clusters * n_feats, sizeof(f64));
	if (!local_sum)
		goto alloc_error;
	f64* count = calloc(n_clusters, sizeof(f64));
	if (!count)
		goto alloc_error;
	f64* sum = calloc(n_clusters * n_feats, sizeof(f64));
	if (!sum)
		goto alloc_error;
	for (usize p = 0; p < len; p++) {
		local_count[labels[p]]++;

		for (usize f = 0; f < n_feats; f++) {
			local_sum[IDX(labels[p], n_feats) + f] +=
				data[IDX(p, n_feats) + f];
		}
	}
	MPI_Allreduce(local_count, count, n_clusters, MPI_DOUBLE, MPI_SUM,
		      MPI_COMM_WORLD);
	MPI_Allreduce(local_sum, sum, n_clusters * n_feats, MPI_DOUBLE, MPI_SUM,
		      MPI_COMM_WORLD);
	for (usize c = 0; c < n_clusters; c++) {
		f64* centroid = &centroids[IDX(c, n_feats)];
		if (count[c] == 0) {
			// do something
			continue;
		}
		for (usize f = 0; f < n_feats; f++) {
			centroid[f] = sum[IDX(c, n_feats) + f] / count[c];
		}
	}
	free(sum);
	free(count);
	free(local_sum);
	free(local_count);

	return;
alloc_error:
	fprintf(stderr, "alloc error\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
}

void kmeans_mpi_fit(struct mpi_ctx* ctx, struct kmeans* km, struct dataset* X)
{
	usize len;
	u32   n_feats;
	f64*  centroids;

	if (ctx->rank == 0) {
		init_centroids(km, X);
		len = X->len;
		n_feats = X->n_feats;
	}

	i32 *sendcounts, *displs, recvcount = 0;
	f64* recvbuf;
	scatter_data(ctx, km->centroids, km->n_clusters, X->data, &len,
		     &centroids, &n_feats, &sendcounts, &displs, &recvbuf,
		     &recvcount);

	i32  local_len = recvcount / n_feats;
	u32* labels = calloc(len, sizeof(u32));
	if (!labels)
		goto alloc_error;

	for (usize i = 0; i < km->max_iter; i++) {
		i32 local_changed = update_labels(labels, n_feats, centroids,
						  km->n_clusters, recvbuf,
						  local_len);
		mpi_update_centroids(centroids, labels, n_feats, km->n_clusters,
				     recvbuf, local_len);
		i32 changed;
		MPI_Allreduce(&local_changed, &changed, 1, MPI_INT, MPI_LOR,
			      MPI_COMM_WORLD);

		if (!changed)
			break;
	}

	return;

alloc_error:
	fprintf(stderr, "alloc error\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
}
#endif /* ifdef MPI_ENABLED */
