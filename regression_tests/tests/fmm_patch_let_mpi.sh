#!/usr/bin/env bash

TEST_ID="fmm_patch_let_mpi"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/fmm_patch_let_mpi"
RUN_DIR_REL="regression_tests/cases/fmm_patch_let_mpi"
CHECK_FUNCTION="check_fmm_patch_let_mpi_case"
RUN_MODE="slurm"
SLURM_NTASKS="4"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="00:10:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
