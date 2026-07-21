#!/usr/bin/env bash

TEST_ID="rayleigh_taylor_mpi"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/rayleigh_taylor_mpi"
RUN_DIR_REL="regression_tests/cases/rayleigh_taylor_mpi"
CHECK_FUNCTION="check_rayleigh_taylor_case"
RUN_MODE="slurm"
SLURM_NTASKS="128"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
