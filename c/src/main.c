#include <omp.h>
#include <stdio.h>

#include <hpc/kmeans.h>

int main(int argc, char** argv)
{
	struct kmeans  km = kmeans_init(3, 300);
	struct dataset d = from_cmdline(argc, argv);

	f64 start = omp_get_wtime();
	{
		kmeans_fit(&km, &d);
	}
	f64 end = omp_get_wtime();

	printf("Elapsed time: %f seconds\n", end - start);

	return 0;
}
