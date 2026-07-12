#!/usr/bin/env bash

TEST_ID="ddmc_mpi_zero_cell"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/ddmc_mpi_zero_cell"
RUN_DIR_REL="regression_tests/cases/ddmc_mpi_zero_cell"
CHECK_FUNCTION="check_ddmc_mpi_zero_cell_case"
RUN_MODE="slurm"
SLURM_NTASKS="8"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="00:20:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
