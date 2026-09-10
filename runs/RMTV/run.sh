#!/usr/bin/env bash
set -euo pipefail
rmtv_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rmtv_root=$(cd -- "$rmtv_dir/../.." && pwd)
cd "$rmtv_dir"
exec "${RMTV_BIN:-$rmtv_root/build/gnuRelease/RMTV/rich}" "$@"
