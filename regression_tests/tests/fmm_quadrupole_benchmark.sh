#!/usr/bin/env bash

TEST_ID="fmm_quadrupole_benchmark"
TAGS="serial benchmark"
BUILD_TEST_NAME="regression_tests/cases/fmm_quadrupole_benchmark"
RUN_DIR_REL="regression_tests/cases/fmm_quadrupole_benchmark"
CHECK_FUNCTION="check_fmm_quadrupole_benchmark_case"
RUN_MODE="direct"
RUN_COMMAND='"${RICH_BIN}"'
