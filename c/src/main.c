#include <mpi.h>

#include <hpc/kmeans.h>
#include <omp.h>
#include <stdio.h>

int main(void)
{
	struct kmeans  km = kmeans_init(3, 300);
	struct dataset d = read_from_file("data/seeds_dataset.txt");

	f64 start = omp_get_wtime();
	{
		kmeans_fit(&km, &d);
	}
	f64 end = omp_get_wtime();

	printf("Elapsed time: %f seconds\n", end - start);

	return 0;
}
