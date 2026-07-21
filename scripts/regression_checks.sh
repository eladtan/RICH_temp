#!/usr/bin/env bash

# Backward-compatible shim. New location:
# regression_tests/lib/regression_checks.sh

set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/regression_tests/lib/regression_checks.sh"
