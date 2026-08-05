#!/usr/bin/env bash

TEST_ID="amr_neighbor_remove"
TAGS="serial mpi"
BUILD_TEST_NAME="regression_tests/cases/amr_neighbor_remove"
RUN_DIR_REL="regression_tests/cases/amr_neighbor_remove"
CHECK_FUNCTION="check_amr_neighbor_remove_case"

SLURM_TIME_LIMIT="00:20:00"
if [[ "${CONFIG}" == *MPI* ]]; then
    RUN_MODE="slurm"
    SLURM_NTASKS="4"
    SLURM_PARTITION="genoa"
    RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
else
    RUN_MODE="direct"
    RUN_COMMAND='"${RICH_BIN}"'
fi
