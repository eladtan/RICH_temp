#!/usr/bin/env bash

TEST_ID="spherical_collapse_hires"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/spherical_collapse_hires"
RUN_DIR_REL="regression_tests/cases/spherical_collapse_hires"
CHECK_FUNCTION="check_spherical_collapse_case"
BUILD_ARGS="--high-res"
RUN_MODE="slurm"
SLURM_NTASKS="256"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
