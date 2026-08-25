#!/bin/bash
#SBATCH --job-name=lane-rad-reduced
#SBATCH --partition=core
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=slurm-lane-rad-reduced-%j.out

set -uo pipefail
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
openmpi_root=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1
export PATH="${openmpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${openmpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
case_dir="${RICH_CASE_DIR:-${SLURM_SUBMIT_DIR}/regression_tests/cases/lane_radiation_shock_individual}"
repo_root="${RICH_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
rich_bin="${RICH_REDUCED_BINARY:-${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI}"
reduced_root="${RICH_REDUCED_ROOT:-${case_dir}/reduced_${SLURM_JOB_ID}}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/software/x86_64/5.14.0/python/3.12.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${reduced_root}"

run_lane() {
  local lane="$1"
  local lane_dir="${reduced_root}/${lane}"
  mkdir -p "${lane_dir}"
  export THUNDER_ARTIFACT_DIR="${lane_dir}"
  export RICH_TEST_POINT_COUNT=32768
  export RICH_TEST_FINAL_TIME=32.0604
  # Keep the reduced initial step close to the production 3.05752e-4 s
  # while retaining a fine enough integer quantum for synchronized coverage.
  export RICH_TEST_INITIAL_BIN="${RICH_TEST_INITIAL_BIN:-33}"
  export RICH_TEST_MAXIMUM_BIN="${RICH_TEST_MAXIMUM_BIN:-60}"
  export RICH_TEST_TIME_QUANTUM="${RICH_TEST_TIME_QUANTUM:-2.8475356730643257e-14}"
  export RICH_TEST_MAX_CELLS=40000
  export RICH_TEST_MG_PRECONDITIONER=cell_block
  export RICH_TEST_AMR_INTERVAL=8.0151
  export RICH_TEST_HISTORY_INTERVAL=8.0151
  export RICH_TEST_PRESCRIBED_AMR=1
  export RICH_TEST_PRESCRIBED_REFINE=512
  export RICH_TEST_PRESCRIBED_DEREFINE=256
  export RICH_TEST_PRESCRIBED_REFINE_TIME=8.0151
  export RICH_TEST_PRESCRIBED_DEREFINE_TIME=16.0302
  unset RICH_QUIET
  export RICH_TEST_PROGRESS_CYCLES="${RICH_TEST_PROGRESS_CYCLES:-100}"
  export RICH_TEST_PROGRESS_WALL_SECONDS="${RICH_TEST_PROGRESS_WALL_SECONDS:-60}"
  export OMP_NUM_THREADS=1
  if [[ "${lane}" == global ]]; then
    unset RICH_INDIVIDUAL_MODE
  else
    export RICH_INDIVIDUAL_MODE="${lane}"
  fi
  cd "${case_dir}"
  set +e
  mpirun -np 8 --bind-to core --map-by ppr:8:node "${rich_bin}" \
    > "${lane_dir}/run.log" 2>&1
  local status=$?
  set -e
  awk '/MG_BICGSTAB_(PROGRESS|RELIABLE_UPDATE|CONVERGENCE|TIMING|TOLERANCE|HISTORICAL_POLICY)|MG_PRECONDITIONER_(SETUP|APPLY)|MG_(SPECTRAL_POSITIVITY|PASSIVE_ROUNDOFF)_REPAIR|MG_COMPTON_|MG_RADIATION_FORCE_|MG_ENERGY_BALANCE_REJECTION|MG_TIMESTEP_LIMIT|MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH|MG_ACTIVE_OWNERSHIP|INDIVIDUAL_(RADIATION_REJECTION|TERMINAL_EVENT_CLAMP)|Individual radiation retries:|Radiation time step ID|kp=|Group number|Total iterations:|Exited BiCGSTAB|Converged at iter|not good end|Negative raw radiation candidate energy/' \
    "${lane_dir}/run.log" > "${lane_dir}/mg_solver_diagnostics.log"
  printf '%s\n' "${status}" > "${lane_dir}/exit_code.txt"
}

if [[ -n "${RICH_REDUCED_LANE:-}" ]]; then
  case "${RICH_REDUCED_LANE}" in
    global|full|full-variable|partial) ;;
    *)
      printf 'Unknown reduced lane: %s\n' "${RICH_REDUCED_LANE}" >&2
      exit 2
      ;;
  esac
  run_lane "${RICH_REDUCED_LANE}"
  exit "$(<"${reduced_root}/${RICH_REDUCED_LANE}/exit_code.txt")"
fi

run_lane global
run_lane full
run_lane full-variable
run_lane partial

/software/x86_64/5.14.0/python/3.12.1/bin/python3 \
  "${case_dir}/compare_campaign.py" \
  --tier reduced \
  --global-dir "${reduced_root}/global" \
  --full-dir "${reduced_root}/full" \
  --full-variable-dir "${reduced_root}/full-variable" \
  --partial-dir "${reduced_root}/partial" \
  --output "${reduced_root}/verdict.txt"
