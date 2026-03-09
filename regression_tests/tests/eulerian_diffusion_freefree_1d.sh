#!/usr/bin/env bash

TEST_ID="eulerian_diffusion_freefree_1d"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/eulerian_diffusion_freefree_1d"
RUN_DIR_REL="regression_tests/cases/eulerian_diffusion_freefree_1d"
CHECK_FUNCTION="check_eulerian_diffusion_freefree_1d_case"
RUN_MODE="slurm"
SLURM_NTASKS="8"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
