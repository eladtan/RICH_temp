#!/bin/bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s global|full-variable|partial\n' "$0" >&2
  exit 2
fi
lane="$1"
case "${lane}" in
  global|full-variable|partial) ;;
  *) printf 'invalid ownership-smoke lane: %s\n' "${lane}" >&2; exit 2 ;;
esac

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../../.." && pwd)"
smoke_root="${RICH_SMOKE_ROOT:?RICH_SMOKE_ROOT must name the smoke directory}"
lane_dir="${smoke_root}/${lane}"
rich_bin="${RICH_SMOKE_BINARY:-${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI}"
tasks_per_node="${RICH_SMOKE_TASKS_PER_NODE:-8}"
mkdir -p "${lane_dir}"

ulimit -c 0
export THUNDER_ARTIFACT_DIR="${lane_dir}"
export RICH_TEST_POINT_COUNT="${RICH_SMOKE_POINT_COUNT:-32768}"
export RICH_TEST_MAX_CYCLES="${RICH_SMOKE_MAX_CYCLES:-1}"
if [[ "${RICH_SMOKE_ENABLE_AMR:-0}" == 1 ]]; then
  unset RICH_TEST_DISABLE_AMR
else
  export RICH_TEST_DISABLE_AMR=1
fi
export RICH_TEST_INITIAL_BIN="${RICH_SMOKE_INITIAL_BIN:-30}"
export RICH_TEST_MAXIMUM_BIN="${RICH_SMOKE_MAXIMUM_BIN:-60}"
export RICH_TEST_TIME_QUANTUM="${RICH_SMOKE_TIME_QUANTUM:-2.8475356730643257e-13}"
if [[ "${lane}" == global ]]; then
  unset RICH_INDIVIDUAL_MODE
else
  export RICH_INDIVIDUAL_MODE="${lane}"
fi
export OMP_NUM_THREADS=1
unset RICH_QUIET RICH_TEST_PRESCRIBED_AMR

{
  printf 'lane %s\n' "${lane}"
  printf 'job_id %s\n' "${SLURM_JOB_ID:-none}"
  printf 'tasks %s\n' "${SLURM_NTASKS:-8}"
  sha256sum "${rich_bin}"
  module list 2>&1
} > "${lane_dir}/run_info.txt"

cd "${case_dir}"
set +e
mpirun -np "${SLURM_NTASKS:-8}" --bind-to core \
  --map-by "ppr:${tasks_per_node}:node" \
  "${rich_bin}" > "${lane_dir}/run.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${lane_dir}/exit_code.txt"
exit "${status}"
