.PHONY: all run run_gpu run_mpi clean test build build_gpu build_mpi

V ?= 0

ifeq ($(V), 0)
	Q := @
	print := @echo
else
	Q :=
	print := @true
endif

CC := clang
GPUCC := nvc
MPICC := mpicc
export OMPI_CC := $(CC)
OMPFLAGS := -fopenmp

CFLAGS := -std=gnu23 -Iinclude -Wall -Wpedantic -Werror -Wextra -Wshadow $(OMPFLAGS)
LDFLAGS := -lm $(OMPFLAGS)

ifdef DEBUG
	CFLAGS += -g -O0 -fsanitize=address,undefined
	LDFLAGS += -fsanitize=address,undefined
else
	CFLAGS += -O3
endif

OUT_DIR := out
BIN := $(OUT_DIR)/kmeans
GPU_BIN := $(OUT_DIR)/kmeans_gpu
MPI_BIN := $(OUT_DIR)/kmeans_mpi
OBJS := src/dataset.o src/algebra.o src/kmeans.o
MPI_OBJS := $(OBJS:%.o=%_mpi.o)
GPU_OBJS := $(OBJS:%.o=%_gpu.o)
DATASET ?= data/seeds_dataset.txt

all: $(BIN) $(GPU_BIN) $(MPI_BIN)

build: $(BIN)

build_gpu: $(GPU_BIN)

build_mpi: $(MPI_BIN)

run: $(BIN)
	./$< $(DATASET)

run_gpu: $(GPU_BIN)
	./$< $(DATASET)

run_mpi: $(MPI_BIN)
	mpirun $< $(DATASET)

clean:
	rm -rf $(OUT_DIR) src/*.o src/*_mpi.o src/*_gpu.o

test: $(BIN) $(MPI_BIN) clean

$(BIN): $(OBJS) src/main.o | $(OUT_DIR)
	$(print) "  LD   $@"
	$(Q)$(CC) $^ $(LDFLAGS) -o $@

$(GPU_BIN): $(GPU_OBJS) src/main_gpu.o | $(OUT_DIR)
	$(print) "  LD   $@"
	$(Q)$(GPUCC) $^ $(LDFLAGS) -o $@

$(MPI_BIN): $(MPI_OBJS) src/main_mpi.o | $(OUT_DIR)
	$(print) "  LD   $@"
	$(Q)$(MPICC) $^ $(LDFLAGS) -o $@

%.o: %.c
	$(print) "  CC   $@"
	$(Q)$(CC) $(CFLAGS) -c $< -o $@

%_gpu.o: %.c
	$(print) "  CC   $@"
	$(Q)$(GPUCC) $(CFLAGS) -DGPU_ENABLED -c $< -o $@

%_mpi.o: %.c
	$(print) "  CC   $@"
	$(Q)$(MPICC) $(CFLAGS) -DMPI_ENABLED -c $< -o $@

src/main_mpi.o: src/main_mpi.c
	$(print) "  CC   $@"
	$(Q)$(MPICC) $(CFLAGS) -DMPI_ENABLED -c $< -o $@

$(OUT_DIR):
	$(Q)mkdir -p $@
