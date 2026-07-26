#!/usr/bin/env bash

TEST_ID="fmm_sparse_rank_waves"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/fmm_sparse_rank_waves"
RUN_DIR_REL="regression_tests/cases/fmm_sparse_rank_waves"
CHECK_FUNCTION="check_fmm_sparse_rank_waves_case"
RUN_MODE="slurm"
SLURM_NTASKS="6"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
SLURM_TIME_LIMIT="00:05:00"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'

# Rank 0 owns a sparse dusting spanning the whole domain while the other five
# ranks own compact clusters in a dense core. That reproduces, in miniature, the
# geometry that made the TDE restart request ~99.9% of all particles as P2P.
# Six ranks is the minimum that gives a meaningful span ratio.
