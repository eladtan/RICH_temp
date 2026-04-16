#!/usr/bin/env bash

TEST_ID="doppler_scatter_mc"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/doppler_scatter_mc"
BUILD_ARGS="--energy_groups_num=100"
RUN_DIR_REL="regression_tests/cases/doppler_scatter_mc"
RUN_MODE="slurm"
SLURM_NTASKS="64"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="01:00:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
CHECK_FUNCTION="check_doppler_scatter_mc_case"
