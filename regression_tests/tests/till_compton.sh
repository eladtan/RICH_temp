#!/usr/bin/env bash

TEST_ID="till_compton"
BUILD_TEST_NAME="regression_tests/cases/till_compton"
BUILD_ARGS="--energy_groups_num=32"
RUN_DIR_REL="regression_tests/cases/till_compton"
RUN_COMMAND='"${ROOT_DIR}/build/${CONFIG}/rich"'
CHECK_FUNCTION="check_till_case"
