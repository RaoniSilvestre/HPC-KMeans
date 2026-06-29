#!/usr/bin/env bash

RESULTS=results.csv
TOTAL_CORES=$(lscpu -p | rg -v '^#' | sort -u -t, -k2 | wc -l)

echo "impl,N,threads,processes,duration" >$RESULTS

fd '\d+.txt' data | while read -r dataset; do
	N=$(echo $dataset | rg -o '\d+')
	for n_threads in $(seq $TOTAL_CORES); do
		export OMP_NUM_THREADS=$n_threads
		duration=$(make run DATASET=$dataset | rg -o '\d+\.\d+')

		echo "omp,$N,$n_threads,1,$duration" | tee -a $RESULTS
	done
done

fd '\d+.txt' data | while read -r dataset; do
	N=$(echo $dataset | rg -o '\d+')

	export OMP_TARGET_OFFLOAD=MANDATORY
	duration=$(make run_gpu DATASET=$dataset | rg -o '\d+\.\d+')

	echo "gpu,$N,1,1,$duration" | tee -a $RESULTS
done

fd '\d+.txt' data | while read -r dataset; do
	N=$(echo $dataset | rg -o '\d+')
	export OMP_NUM_THREADS=1
	for n_cores in $(seq $TOTAL_CORES); do
		duration=$(mpirun -n $n_cores ./out/kmeans_mpi $dataset </dev/null | rg -o '\d+\.\d+')

		echo "mpi,$N,1,$n_cores,$duration" | tee -a $RESULTS
	done
done

fd '\d+.txt' data | while read -r dataset; do
	N=$(echo $dataset | rg -o '\d+')
	for n_cores in $(seq $TOTAL_CORES); do
		n_threads=$(($TOTAL_CORES / $n_cores))

		export OMP_NUM_THREADS=$n_threads
		duration=$(mpirun -n $n_cores ./out/kmeans_mpi $dataset </dev/null | rg -o '\d+\.\d+')

		echo "mpi_omp,$N,$n_threads,$n_cores,$duration" | tee -a $RESULTS
	done
done
