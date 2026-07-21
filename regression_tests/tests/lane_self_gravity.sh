#!/usr/bin/env bash

TEST_ID="lane_self_gravity"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/lane_self_gravity"
RUN_DIR_REL="regression_tests/cases/lane_self_gravity"
CHECK_FUNCTION="check_lane_self_gravity_case"
RUN_MODE="slurm"
SLURM_NTASKS="64"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
