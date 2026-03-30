#!/usr/bin/env bash

TEST_ID="yee_vortex_128"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/yee_vortex_128"
RUN_DIR_REL="regression_tests/cases/yee_vortex_128"
CHECK_FUNCTION="check_yee_vortex_case"
RUN_MODE="slurm"
SLURM_NTASKS="16"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
