#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s scalar|cell_block\n' "$0" >&2
  exit 2
fi

preconditioner="$1"
case "${preconditioner}" in
  scalar|cell_block) ;;
  *) printf 'invalid preconditioner: %s\n' "${preconditioner}" >&2; exit 2 ;;
esac

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../../.." && pwd)"
benchmark_dir="${repo_root}/regression_tests/cases/lane_radiation_shock_individual"
campaign_root="${RICH_PRECONDITIONER_AB_ROOT:?RICH_PRECONDITIONER_AB_ROOT must name the shared A/B directory}"
lane_dir="${campaign_root}/${preconditioner}"
rich_bin="${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI"
mkdir -p "${lane_dir}"

if [[ ! -x "${rich_bin}" ]]; then
  printf 'missing executable: %s\n' "${rich_bin}" > "${lane_dir}/driver_error.txt"
  printf '127\n' > "${lane_dir}/exit_code.txt"
  exit 127
fi

{
  printf 'preconditioner %s\n' "${preconditioner}"
  printf 'job_id %s\n' "${SLURM_JOB_ID:-none}"
  printf 'nodes %s\n' "${SLURM_JOB_NUM_NODES:-unknown}"
  printf 'tasks %s\n' "${SLURM_NTASKS:-1}"
  printf 'point_count %s\n' "${RICH_AB_POINT_COUNT:?RICH_AB_POINT_COUNT is required}"
  printf 'dt %s\n' "${RICH_AB_DT:?RICH_AB_DT is required}"
  printf 'git_revision '
  git -C "${repo_root}" rev-parse HEAD
  sha256sum "${rich_bin}"
  module list 2>&1
} > "${lane_dir}/run_info.txt"

export THUNDER_ARTIFACT_DIR="${lane_dir}"
export RICH_TEST_POINT_COUNT="${RICH_AB_POINT_COUNT}"
export RICH_TEST_FINAL_TIME="${RICH_AB_DT}"
export RICH_TEST_INITIAL_DT="${RICH_AB_DT}"
export RICH_TEST_INITIAL_BIN=40
export RICH_TEST_MAXIMUM_BIN=40
export RICH_TEST_DISABLE_ROUNDING=1
export RICH_TEST_DISABLE_AMR=1
export RICH_TEST_MG_PRECONDITIONER="${preconditioner}"
export RICH_TEST_PROGRESS_CYCLES=100
export RICH_TEST_PROGRESS_WALL_SECONDS=60
export RICH_MG_DISTRIBUTED_ACTIVE_PROFILE=1
export OMP_NUM_THREADS=1
unset RICH_INDIVIDUAL_MODE RICH_QUIET

cd "${benchmark_dir}"
set +e
mpirun -np "${SLURM_NTASKS:-1}" --bind-to core \
  --map-by "ppr:${RICH_AB_PPR:-1}:node" \
  "${rich_bin}" > "${lane_dir}/run.log" 2>&1
status=$?
set -e

awk '/MG_PRECONDITIONER|MG_BICGSTAB_(PROGRESS|CONVERGENCE|TIMING)|MG_TIMESTEP_LIMIT|Radiation time step ID|kp=|Group number|Total iterations:|Exited BiCGSTAB|Converged at iter|not good end|Negative raw radiation candidate energy/' \
  "${lane_dir}/run.log" > "${lane_dir}/mg_solver_diagnostics.log"
printf '%s\n' "${status}" > "${lane_dir}/exit_code.txt"
exit "${status}"
