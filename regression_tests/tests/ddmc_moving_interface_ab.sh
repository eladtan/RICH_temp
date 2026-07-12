#!/usr/bin/env bash

TEST_ID="ddmc_moving_interface_ab"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/ddmc_moving_interface_ab"
RUN_DIR_REL="regression_tests/cases/ddmc_moving_interface_ab"
CHECK_FUNCTION="check_ddmc_moving_interface_ab_case"
RUN_MODE="slurm"
SLURM_NTASKS="1"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="00:20:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
