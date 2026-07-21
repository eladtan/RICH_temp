#!/usr/bin/env bash

TEST_ID="spherical_collapse"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/spherical_collapse"
RUN_DIR_REL="regression_tests/cases/spherical_collapse"
CHECK_FUNCTION="check_spherical_collapse_case"
RUN_MODE="slurm"
SLURM_NTASKS="64"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
