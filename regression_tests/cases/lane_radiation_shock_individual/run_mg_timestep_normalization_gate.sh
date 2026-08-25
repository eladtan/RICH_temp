#!/bin/bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../../.." && pwd)"
gate_root="${RICH_MG_TIMESTEP_GATE_ROOT:?RICH_MG_TIMESTEP_GATE_ROOT must be set}"
rich_bin="${RICH_MG_TIMESTEP_GATE_BINARY:-${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI}"
python_bin=/software/x86_64/5.14.0/python/3.12.1/bin/python3

mkdir -p "${gate_root}"
sha256sum "${rich_bin}" > "${gate_root}/binary.sha256"
cd "${case_dir}"

if [[ "${RICH_MG_TIMESTEP_GATE_EMPTY_ONLY:-0}" != 1 ]]; then
for lane in global full full-reduced full-variable partial; do
  lane_dir="${gate_root}/${lane}"
  mkdir -p "${lane_dir}"
  export THUNDER_ARTIFACT_DIR="${lane_dir}"
  export RICH_TEST_POINT_COUNT="${RICH_MG_TIMESTEP_GATE_POINTS:-32768}"
  export RICH_TEST_INITIAL_BIN=30
  export RICH_TEST_MAXIMUM_BIN=60
  export RICH_TEST_TIME_QUANTUM=2.8475356730643257e-13
  export RICH_TEST_MAX_CYCLES=1
  export RICH_TEST_MAX_CELLS=65536
  export RICH_TEST_DISABLE_AMR=1
  export RICH_TEST_PROGRESS_CYCLES=1
  export RICH_TEST_PROGRESS_WALL_SECONDS=1
  export OMP_NUM_THREADS=1
  unset RICH_QUIET RICH_TEST_SPARSE_INITIAL_BIN \
    RICH_TEST_SPARSE_SINGLE_RANK RICH_TEST_FORCE_CENTRAL_POINT \
    RICH_TEST_REQUIRE_EMPTY_OWNED_RANK RICH_TEST_DISABLE_GRAVITY \
    RICH_TEST_ENABLE_GREY_DIFFUSION_FORCE RICH_TEST_SPARSE_MAX_ER_CELL \
    RICH_TEST_FORCE_REDUCED_ALL_ACTIVE \
    RICH_TEST_GRAVITY_PREDICTOR_SPEED \
    RICH_MG_DISTRIBUTED_ACTIVE_PROFILE

  case "${lane}" in
    global)
      unset RICH_INDIVIDUAL_MODE
      ;;
    full)
      export RICH_INDIVIDUAL_MODE=full
      ;;
    full-reduced)
      export RICH_INDIVIDUAL_MODE=full
      export RICH_TEST_FORCE_REDUCED_ALL_ACTIVE=1
      ;;
    full-variable|partial)
      export RICH_INDIVIDUAL_MODE="${lane}"
      export RICH_MG_DISTRIBUTED_ACTIVE_PROFILE=1
      export RICH_TEST_SPARSE_INITIAL_BIN=1
      export RICH_TEST_SPARSE_SINGLE_RANK=1
      export RICH_TEST_SPARSE_MAX_ER_CELL=1
      export RICH_TEST_GRAVITY_PREDICTOR_SPEED=1e7
      ;;
  esac

  set +e
  if [[ "${RICH_MG_TIMESTEP_GATE_DIRECT:-0}" == 1 ]]; then
    "${rich_bin}" > "${lane_dir}/run.log" 2>&1
  else
    mpirun -np "${SLURM_NTASKS:-128}" --bind-to core --map-by ppr:16:node \
      "${rich_bin}" > "${lane_dir}/run.log" 2>&1
  fi
  status=$?
  set -e
  printf '%s\n' "${status}" > "${lane_dir}/exit_code.txt"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
done
fi

lane_dir="${gate_root}/empty-owned"
mkdir -p "${lane_dir}"
export THUNDER_ARTIFACT_DIR="${lane_dir}"
export RICH_TEST_POINT_COUNT="${RICH_MG_TIMESTEP_EMPTY_OWNED_POINTS:-64}"
export RICH_TEST_INITIAL_BIN=30
export RICH_TEST_MAXIMUM_BIN=60
export RICH_TEST_TIME_QUANTUM=2.8475356730643257e-13
export RICH_TEST_MAX_CYCLES=1
export RICH_TEST_MAX_CELLS=64
export RICH_TEST_DISABLE_AMR=1
export RICH_TEST_PROGRESS_CYCLES=1
export RICH_TEST_PROGRESS_WALL_SECONDS=1
export RICH_TEST_FORCE_CENTRAL_POINT=1
export RICH_TEST_REQUIRE_EMPTY_OWNED_RANK=1
export RICH_TEST_DISABLE_GRAVITY=1
export RICH_TEST_ENABLE_GREY_DIFFUSION_FORCE=1
export RICH_INDIVIDUAL_MODE=full
export OMP_NUM_THREADS=1
unset RICH_QUIET RICH_TEST_SPARSE_INITIAL_BIN \
  RICH_TEST_SPARSE_SINGLE_RANK RICH_MG_DISTRIBUTED_ACTIVE_PROFILE
set +e
if [[ "${RICH_MG_TIMESTEP_GATE_DIRECT:-0}" == 1 ]]; then
  "${rich_bin}" > "${lane_dir}/run.log" 2>&1
else
  mpirun -np "${SLURM_NTASKS:-128}" --bind-to core --map-by ppr:16:node \
    "${rich_bin}" > "${lane_dir}/run.log" 2>&1
fi
status=$?
set -e
printf '%s\n' "${status}" > "${lane_dir}/exit_code.txt"
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

if [[ "${RICH_MG_TIMESTEP_GATE_EMPTY_ONLY:-0}" == 1 ]]; then
  exit 0
fi

run_restart_lane() {
  local name="$1"
  local cycles="$2"
  local restart_input="$3"
  local restart_dir="${gate_root}/${name}"
  mkdir -p "${restart_dir}"
  export THUNDER_ARTIFACT_DIR="${restart_dir}"
  export RICH_TEST_POINT_COUNT="${RICH_MG_RESTART_GATE_POINTS:-512}"
  export RICH_TEST_INITIAL_BIN=30
  export RICH_TEST_MAXIMUM_BIN=60
  export RICH_TEST_TIME_QUANTUM=2.8475356730643257e-13
  export RICH_TEST_MAX_CYCLES="${cycles}"
  export RICH_TEST_MAX_CELLS=1024
  export RICH_TEST_DISABLE_AMR=1
  export RICH_TEST_DISABLE_GRAVITY=1
  export RICH_TEST_PROGRESS_CYCLES=1
  export RICH_TEST_PROGRESS_WALL_SECONDS=1
  export RICH_INDIVIDUAL_MODE=full
  export OMP_NUM_THREADS=1
  unset RICH_QUIET RICH_TEST_SPARSE_INITIAL_BIN \
    RICH_TEST_SPARSE_SINGLE_RANK RICH_TEST_SPARSE_MAX_ER_CELL \
    RICH_TEST_FORCE_CENTRAL_POINT RICH_TEST_REQUIRE_EMPTY_OWNED_RANK \
    RICH_TEST_ENABLE_GREY_DIFFUSION_FORCE RICH_TEST_FINAL_TIME
  if [[ -n "${restart_input}" ]]; then
    export RICH_TEST_RESTART_INPUT="${restart_input}"
  else
    unset RICH_TEST_RESTART_INPUT
  fi

  set +e
  if [[ "${RICH_MG_TIMESTEP_GATE_DIRECT:-0}" == 1 ]]; then
    "${rich_bin}" > "${restart_dir}/run.log" 2>&1
  else
    mpirun -np "${SLURM_NTASKS:-128}" --bind-to core --map-by ppr:16:node \
      "${rich_bin}" > "${restart_dir}/run.log" 2>&1
  fi
  local restart_status=$?
  set -e
  printf '%s\n' "${restart_status}" > "${restart_dir}/exit_code.txt"
  if [[ "${restart_status}" -ne 0 ]]; then
    exit "${restart_status}"
  fi
}

run_restart_lane restart-source 1 ""
run_restart_lane restart-continuous 2 ""
run_restart_lane restart-resumed 2 \
  "${gate_root}/restart-source/final_state.h5"

adverse_pattern='MG_BICGSTAB_(CONVERGENCE|RESULT|TIMING).*outcome=(rejected|breakdown|not_converged|nonfinite)|MG_MPI_FATAL|Reducing dt|INDIVIDUAL_RADIATION_REJECTION'
mapfile -d '' run_logs < <(find "${gate_root}" -name run.log -type f -print0)
if grep -EnH "${adverse_pattern}" "${run_logs[@]}" \
    > "${gate_root}/adverse_events.txt"; then
  exit 1
fi
rm -f "${gate_root}/adverse_events.txt"

"${python_bin}" "${case_dir}/compare_mg_timestep_normalization.py" \
  --global-log "${gate_root}/global/run.log" \
  --full-log "${gate_root}/full/run.log" \
  --forced-reduced-log "${gate_root}/full-reduced/run.log" \
  --full-variable-log "${gate_root}/full-variable/run.log" \
  --partial-log "${gate_root}/partial/run.log" \
  --empty-owned-log "${gate_root}/empty-owned/run.log" \
  --restart-source-log "${gate_root}/restart-source/run.log" \
  --restart-resumed-log "${gate_root}/restart-resumed/run.log" \
  --restart-continuous-log "${gate_root}/restart-continuous/run.log" \
  | tee "${gate_root}/verdict.txt"
