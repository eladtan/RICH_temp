#!/usr/bin/env bash

TEST_ID="mach2_diffusion"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/mach2_diffusion"
RUN_DIR_REL="regression_tests/cases/mach2_diffusion"
CHECK_FUNCTION="check_mach2_case"
RUN_MODE="slurm"
SLURM_NTASKS="8"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
