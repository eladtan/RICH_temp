#!/usr/bin/env bash

TEST_ID="fmm_mpi_scaling_benchmark"
TAGS="mpi manual benchmark"
BUILD_TEST_NAME="regression_tests/cases/fmm_mpi_scaling_benchmark"
RUN_DIR_REL="regression_tests/cases/fmm_mpi_scaling_benchmark"
CHECK_FUNCTION="check_fmm_mpi_scaling_benchmark_case"
RUN_MODE="slurm"
FMM_MPI_BENCH_RANKS_PER_NODE="${FMM_MPI_BENCH_RANKS_PER_NODE:-4}"
FMM_MPI_BENCH_MAX_NODES="${FMM_MPI_BENCH_MAX_NODES:-16}"
SLURM_NTASKS="$((FMM_MPI_BENCH_MAX_NODES * FMM_MPI_BENCH_RANKS_PER_NODE))"
SLURM_NODES="${FMM_MPI_BENCH_MAX_NODES}"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="08:00:00"
RUN_COMMAND='"${ROOT_DIR}/regression_tests/cases/fmm_mpi_scaling_benchmark/run_benchmark.sh" "${RICH_BIN}"'

# Manual-only. The default remains the original 8/16-node matrix. Set
# FMM_MPI_BENCH_MAX_NODES=64 and FMM_MPI_BENCH_EXTRA_NODES="32 64" to add
# 10-million-particle runs at 32 and 64 nodes while preserving the original
# regression outputs and checker.
