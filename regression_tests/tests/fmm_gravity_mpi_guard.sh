#!/usr/bin/env bash

TEST_ID="fmm_gravity_mpi_guard"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/fmm_gravity_mpi_guard"
RUN_DIR_REL="regression_tests/cases/fmm_gravity_mpi_guard"
CHECK_FUNCTION="check_fmm_gravity_mpi_guard_case"
RUN_MODE="slurm"
SLURM_NTASKS="4"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="00:05:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
