#!/usr/bin/env bash

TEST_ID="gresho_lagrangian"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/gresho_lagrangian"
RUN_DIR_REL="regression_tests/cases/gresho_lagrangian"
CHECK_FUNCTION="check_gresho_case"
RUN_MODE="slurm"
SLURM_NTASKS="8"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
