#!/usr/bin/env bash

TEST_ID="amr_distributed_clip"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/amr_distributed_clip"
RUN_DIR_REL="regression_tests/cases/amr_distributed_clip"
CHECK_FUNCTION="check_amr_distributed_clip_case"

SLURM_TIME_LIMIT="01:00:00"
RUN_MODE="slurm"
SLURM_NTASKS="64"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
