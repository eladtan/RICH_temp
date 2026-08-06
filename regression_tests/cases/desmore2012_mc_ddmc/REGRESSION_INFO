#!/usr/bin/env bash

TEST_ID="desmore2012_mc_ddmc"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/desmore2012_mc_ddmc"
BUILD_ARGS="--energy_groups_num=30"
RUN_DIR_REL="regression_tests/cases/desmore2012_mc_ddmc"
CHECK_FUNCTION="check_desmore2012_mc_ddmc_case"
RUN_MODE="slurm"
SLURM_NTASKS="32"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="02:00:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
