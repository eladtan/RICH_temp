#!/usr/bin/env bash

TEST_ID="doppler_mc"
TAGS="serial"
BUILD_TEST_NAME="regression_tests/cases/doppler_mc"
BUILD_ARGS="--energy_groups_num=100"
RUN_DIR_REL="regression_tests/cases/doppler_mc"
RUN_COMMAND='"${RICH_BIN}"'
CHECK_FUNCTION="check_doppler_mc_case"
SLURM_TIME_LIMIT="00:30:00"
