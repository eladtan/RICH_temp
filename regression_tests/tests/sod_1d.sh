#!/usr/bin/env bash

TEST_ID="sod_1d"
BUILD_TEST_NAME="regression_tests/cases/sod_1d"
RUN_DIR_REL="regression_tests/cases/sod_1d"
RUN_COMMAND='"${ROOT_DIR}/build/${CONFIG}/rich"'
CHECK_FUNCTION="check_sod_case"
