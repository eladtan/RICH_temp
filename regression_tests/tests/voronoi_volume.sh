#!/usr/bin/env bash

TEST_ID="voronoi_volume"
TAGS="serial mpi"
BUILD_TEST_NAME="regression_tests/cases/voronoi_volume"
RUN_DIR_REL="regression_tests/cases/voronoi_volume"
CHECK_FUNCTION="check_voronoi_volume_case"

if [[ "${CONFIG}" == *MPI* ]]; then
    RUN_MODE="slurm"
    SLURM_NTASKS="64"
    SLURM_PARTITION="bigrun"
    SLURM_EXCLUSIVE="1"
    RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
else
    RUN_MODE="direct"
    RUN_COMMAND='"${RICH_BIN}"'
fi
