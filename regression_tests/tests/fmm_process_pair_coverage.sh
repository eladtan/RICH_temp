#!/usr/bin/env bash

TEST_ID="fmm_process_pair_coverage"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/fmm_process_pair_coverage"
RUN_DIR_REL="regression_tests/cases/fmm_process_pair_coverage"
CHECK_FUNCTION="check_fmm_process_pair_coverage_case"
RUN_MODE="slurm"
SLURM_NTASKS="7"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="00:05:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'

# Seven ranks deliberately exercise non-power-of-two process-tree coverage.
