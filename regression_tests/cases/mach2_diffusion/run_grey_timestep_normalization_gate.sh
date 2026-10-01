#!/bin/bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gate_root="${RICH_GREY_TIMESTEP_GATE_ROOT:?RICH_GREY_TIMESTEP_GATE_ROOT must be set}"
rich_bin="${RICH_GREY_TIMESTEP_GATE_BINARY:?RICH_GREY_TIMESTEP_GATE_BINARY must be set}"
python_bin="${RICH_GREY_TIMESTEP_GATE_PYTHON:-/software/x86_64/5.14.0/python/3.12.1/bin/python3}"
tasks="${SLURM_NTASKS:-16}"

mkdir -p "${gate_root}"

run_lane() {
  local lane="$1"
  local points="$2"
  local mode="$3"
  local sparse="$4"
  local lane_dir="${gate_root}/${lane}"
  mkdir -p "${lane_dir}"
  export THUNDER_ARTIFACT_DIR="${lane_dir}"
  export RICH_TEST_POINT_COUNT="${points}"
  export RICH_TEST_MAX_CYCLES=1
  export OMP_NUM_THREADS=1
  export RICH_RUNTIME_LOG=detailed
  unset RICH_INDIVIDUAL_MODE RICH_TEST_SPARSE_INITIAL_BIN \
    RICH_TEST_SPARSE_MAX_ER_CELL
  if [[ -n "${mode}" ]]; then
    export RICH_INDIVIDUAL_MODE="${mode}"
  fi
  if [[ "${sparse}" == "1" ]]; then
    export RICH_TEST_SPARSE_INITIAL_BIN=1
    export RICH_TEST_SPARSE_MAX_ER_CELL=1
  fi

  set +e
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    mpirun -np "${tasks}" --bind-to core --map-by ppr:16:node \
      "${rich_bin}" > "${lane_dir}/run.log" 2>&1
  else
    mpirun -np "${tasks}" "${rich_bin}" \
      > "${lane_dir}/run.log" 2>&1
  fi
  local status=$?
  set -e
  printf '%s\n' "${status}" > "${lane_dir}/exit_code.txt"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

run_lane global 1024 "" 0
run_lane global-empty-owned 8 "" 0
run_lane full 1024 full 0
run_lane full-variable 1024 full-variable 1
run_lane partial 1024 partial 1
run_lane empty-owned 8 full 0

export LD_LIBRARY_PATH="/software/x86_64/5.14.0/python/3.12.1/lib:/software/x86_64/5.14.0/python/3.12.1/lib/python3.12/site-packages/h5py.libs:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
"${python_bin}" "${case_dir}/compare_grey_timestep_normalization.py" \
  --global-log "${gate_root}/global/run.log" \
  --global-empty-owned-log "${gate_root}/global-empty-owned/run.log" \
  --full-log "${gate_root}/full/run.log" \
  --full-variable-log "${gate_root}/full-variable/run.log" \
  --partial-log "${gate_root}/partial/run.log" \
  --empty-owned-log "${gate_root}/empty-owned/run.log" \
  | tee "${gate_root}/verdict.txt"
