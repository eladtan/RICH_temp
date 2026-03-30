#!/usr/bin/env bash

TEST_ID="eulerian_diffusion_freefree_suite"
TAGS="mpi"
BUILD_TEST_NAME="regression_tests/cases/eulerian_diffusion_freefree_1d"
RUN_DIR_REL="regression_tests/cases/eulerian_diffusion_freefree_1d"
CHECK_FUNCTION="check_eulerian_diffusion_freefree_suite_case"
RUN_MODE="direct"
RUN_COMMAND='bash "${ROOT_DIR}/regression_tests/run_freefree_512_32.sh" "${CONFIG}"'
