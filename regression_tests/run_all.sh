#!/usr/bin/env bash

set -euo pipefail

RICH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THUNDER_ROOT="${RICH_ROOT}/regression_tests/THUNDER"

# Match build_rich.sh: ambient CPATH entries can outrank dependency include
# directories selected by CMake.  Clearing it keeps regression builds
# reproducible across module systems and MPI implementations.
unset CPATH

exec "${THUNDER_ROOT}/run_all.sh" \
  --thunder-config "${RICH_ROOT}/regression_tests/config.json" "$@"
