#!/usr/bin/env bash

TEST_ID="sedov_3d_mpi"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/sedov_3d_mpi"
RUN_DIR_REL="regression_tests/cases/sedov_3d_mpi"
CHECK_FUNCTION="check_sedov_case"
RUN_MODE="slurm"
SLURM_NTASKS="128"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
