.PHONY: all run clean test build

CC := gcc
MPICC := mpicc
CCFLAGS := -lm -Iinclude -Wall -Wpedantic -Werror -Wextra -Wshadow -fopenmp
OUT_DIR := out
FILES := src/dataset.c src/algebra.c src/kmeans.c

EXECUTABLE := $(OUT_DIR)/kmeans
MPI_EXECUTABLE := $(OUT_DIR)/kmeans_mpi

all: build build_mpi

run: build
	exec $(EXECUTABLE)

run_mpi: build_mpi
	mpirun $(MPI_EXECUTABLE)

build:
	mkdir -p $(OUT_DIR)
	$(CC) $(CCFLAGS) $(FILES) src/main.c -O3 -o $(EXECUTABLE)

build_mpi:
	mkdir -p $(OUT_DIR)
	$(MPICC) $(CCFLAGS) $(FILES) src/main_mpi.c -O3 -DMPI_ENABLED -o $(MPI_EXECUTABLE)

clean:
	rm -rf $(OUT_DIR)

test: build_mpi build clean
