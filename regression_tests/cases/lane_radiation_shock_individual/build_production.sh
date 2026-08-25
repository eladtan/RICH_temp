#!/bin/bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../../.." && pwd)"

ml openmpi/4.1.6/Intel/OneApi/2024.2.1
openmpi_root=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1
export PATH="${openmpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${openmpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_root}"
./build_rich.sh intelReleaseMPI \
  --test_name=regression_tests/cases/lane_radiation_shock_individual \
  --energy_groups_num=16

test -x "${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI"
sha256sum "${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI"
