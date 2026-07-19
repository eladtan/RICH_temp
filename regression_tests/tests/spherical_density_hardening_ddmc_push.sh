#!/usr/bin/env bash

TEST_ID="spherical_density_hardening_ddmc_push"
TAGS="mpi radiation hydro ddmc multigroup 3d"
BUILD_TEST_NAME="regression_tests/cases/spherical_density_hardening_ddmc_push"
BUILD_ARGS="--energy_groups_num=32"
RUN_DIR_REL="regression_tests/cases/spherical_density_hardening_ddmc_push"
CHECK_FUNCTION="check_spherical_density_hardening_ddmc_push_case"

RUN_MODE="slurm"
SLURM_NTASKS=96
SLURM_NODES=8
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE=1
SLURM_TIME_LIMIT="06:00:00"

RUN_COMMAND='mpirun -np ${SLURM_NTASKS} --map-by node "${RICH_BIN}"'
