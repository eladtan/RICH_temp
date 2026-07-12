#!/usr/bin/env bash

TEST_ID="fmm_mpi_scaling_benchmark"
TAGS="mpi manual benchmark"
BUILD_TEST_NAME="regression_tests/cases/fmm_mpi_scaling_benchmark"
RUN_DIR_REL="regression_tests/cases/fmm_mpi_scaling_benchmark"
CHECK_FUNCTION="check_fmm_mpi_scaling_benchmark_case"
RUN_MODE="slurm"
FMM_MPI_BENCH_RANKS_PER_NODE="${FMM_MPI_BENCH_RANKS_PER_NODE:-4}"
SLURM_NTASKS="$((16 * FMM_MPI_BENCH_RANKS_PER_NODE))"
SLURM_NODES="16"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="08:00:00"
RUN_COMMAND='"${ROOT_DIR}/regression_tests/cases/fmm_mpi_scaling_benchmark/run_benchmark.sh" "${RICH_BIN}"'

# Manual-only. The memory-safe default is 4 MPI ranks per node: 32 ranks on
# 8 nodes and 64 ranks on 16 nodes. Increase FMM_MPI_BENCH_RANKS_PER_NODE only
# after confirming per-node MaxRSS for the 1e6-particle case.
