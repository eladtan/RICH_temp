#!/bin/bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../../.." && pwd)"
gate_root="${RICH_PARTIAL_AMR_GATE_ROOT:?RICH_PARTIAL_AMR_GATE_ROOT must be set}"
rich_bin="${RICH_PARTIAL_AMR_GATE_BINARY:-${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI}"

mkdir -p "${gate_root}"
sha256sum "${rich_bin}" > "${gate_root}/binary.sha256"
export THUNDER_ARTIFACT_DIR="${gate_root}/partial"
mkdir -p "${THUNDER_ARTIFACT_DIR}"
export RICH_TEST_POINT_COUNT="${RICH_PARTIAL_AMR_GATE_POINTS:-8192}"
export RICH_TEST_INITIAL_BIN=30
export RICH_TEST_MAXIMUM_BIN=60
export RICH_TEST_TIME_QUANTUM=2.8475356730643257e-13
export RICH_TEST_MAX_CYCLES="${RICH_PARTIAL_AMR_GATE_CYCLES:-4}"
export RICH_TEST_MAX_CELLS=20000
export RICH_TEST_AMR_BATCH=8192
export RICH_TEST_AMR_INTERVAL=1e-5
export RICH_TEST_PRESCRIBED_AMR=1
export RICH_TEST_PRESCRIBED_REFINE=8192
export RICH_TEST_PRESCRIBED_DEREFINE=0
export RICH_TEST_PRESCRIBED_REFINE_TIME=0
export RICH_TEST_PRESCRIBED_DEREFINE_TIME=1e300
export RICH_TEST_SPARSE_INITIAL_BIN=1
export RICH_TEST_SPARSE_SINGLE_RANK=1
export RICH_TEST_SPARSE_MAX_ER_CELL=1
export RICH_TEST_PROGRESS_CYCLES=1
export RICH_TEST_PROGRESS_WALL_SECONDS=1
export RICH_INDIVIDUAL_MODE=partial
export OMP_NUM_THREADS=1
unset RICH_QUIET RICH_TEST_DISABLE_AMR RICH_TEST_DISABLE_GRAVITY \
  RICH_TEST_REQUIRE_EMPTY_OWNED_RANK RICH_TEST_ENABLE_GREY_DIFFUSION_FORCE \
  RICH_TEST_RESTART_INPUT RICH_TEST_FINAL_TIME

cd "${case_dir}"
set +e
mpirun -np "${SLURM_NTASKS:-128}" --bind-to core --map-by ppr:16:node \
  "${rich_bin}" > "${THUNDER_ARTIFACT_DIR}/run.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${THUNDER_ARTIFACT_DIR}/exit_code.txt"
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

adverse_pattern='MG_BICGSTAB_(CONVERGENCE|RESULT|TIMING).*outcome=(rejected|breakdown|not_converged|nonfinite)|MG_MPI_FATAL|Reducing dt|INDIVIDUAL_RADIATION_REJECTION'
if grep -Eq "${adverse_pattern}" "${THUNDER_ARTIFACT_DIR}/run.log"; then
  grep -En "${adverse_pattern}" "${THUNDER_ARTIFACT_DIR}/run.log" \
    > "${gate_root}/adverse_events.txt"
  exit 1
fi

grep -Eq 'Removing [0-9]+ cells and refining [1-9][0-9]* cells' \
  "${THUNDER_ARTIFACT_DIR}/run.log"
test -f "${THUNDER_ARTIFACT_DIR}/final_state.h5"
printf '%s\n' \
  'PASS: distributed partial AMR completed after sparse nonidentity mesh events' \
  > "${gate_root}/verdict.txt"
