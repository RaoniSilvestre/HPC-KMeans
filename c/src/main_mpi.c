#include <mpi.h>
#include <stdio.h>

#include <hpc/kmeans.h>

int main(int argc, char** argv)
{
	struct mpi_ctx ctx;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ctx.num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);

	struct dataset d;
	struct kmeans  km = kmeans_init(3, 300);
	if (ctx.rank == 0) {
		d = from_cmdline(argc, argv);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	f64 start = MPI_Wtime();

	kmeans_mpi_fit(&ctx, &km, &d);

	MPI_Barrier(MPI_COMM_WORLD);
	f64 end = MPI_Wtime();

	if (ctx.rank == 0) {
		printf("Elapsed time: %f seconds\n", end - start);
	}

	MPI_Finalize();

	return 0;
}
