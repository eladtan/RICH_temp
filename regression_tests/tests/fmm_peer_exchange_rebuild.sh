#!/usr/bin/env bash

TEST_ID="fmm_peer_exchange_rebuild"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/fmm_peer_exchange_rebuild"
RUN_DIR_REL="regression_tests/cases/fmm_peer_exchange_rebuild"
CHECK_FUNCTION="check_fmm_peer_exchange_rebuild_case"
RUN_MODE="slurm"
SLURM_NTASKS="7"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="00:05:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'

# Repeatedly changes isolated and connected graph roles on a persistent
# FmmPeerExchange.  Seven ranks also exercises non-power-of-two layouts.
