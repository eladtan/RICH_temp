#!/usr/bin/env bash

set -euo pipefail

RICH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THUNDER_ROOT="${RICH_ROOT}/regression_tests/THUNDER"
exec "${THUNDER_ROOT}/run_all.sh" \
  --thunder-config "${RICH_ROOT}/regression_tests/config.json" "$@"
