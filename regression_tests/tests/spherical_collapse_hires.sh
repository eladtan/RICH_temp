#!/usr/bin/env bash

TEST_ID="spherical_collapse_hires"
TAGS="mpi manual"
BUILD_TEST_NAME="regression_tests/cases/spherical_collapse_hires"
BUILD_ARGS="--high-res"
RUN_DIR_REL="regression_tests/cases/spherical_collapse_hires"
CHECK_FUNCTION="check_spherical_collapse_case"
RUN_MODE="slurm"
SLURM_NTASKS="128"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="06:00:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
