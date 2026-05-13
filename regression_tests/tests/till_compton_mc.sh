#!/usr/bin/env bash

TEST_ID="till_compton_mc"
TAGS="serial mc"
BUILD_TEST_NAME="regression_tests/cases/till_compton_mc"
BUILD_ARGS="--energy_groups_num=32"
RUN_DIR_REL="regression_tests/cases/till_compton_mc"
RUN_COMMAND='"${RICH_BIN}"'
CHECK_FUNCTION="check_till_mc_case"
SLURM_TIME_LIMIT="02:00:00"
