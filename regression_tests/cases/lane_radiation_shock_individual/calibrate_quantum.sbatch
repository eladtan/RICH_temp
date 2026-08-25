#!/bin/bash
#SBATCH --job-name=lane-rad-calibrate
#SBATCH --partition=core
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm-lane-rad-calibrate-%j.out

set -uo pipefail
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
openmpi_root=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1
export PATH="${openmpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${openmpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
case_dir="${RICH_CASE_DIR:?RICH_CASE_DIR must name the benchmark case directory}"
repo_root="${RICH_REPO_ROOT:?RICH_REPO_ROOT must name the repository root}"
campaign_root="${RICH_CAMPAIGN_ROOT:?RICH_CAMPAIGN_ROOT must name the shared campaign directory}"
rich_bin="${RICH_CAMPAIGN_BINARY:-${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI}"
calibration_root="${campaign_root}/calibration"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/software/x86_64/5.14.0/python/3.12.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${calibration_root}/dt" "${calibration_root}/dt_half"

run_calibration() {
  local label="$1"
  local initial_bin="$2"
  local output_dir="${calibration_root}/${label}"
  export THUNDER_ARTIFACT_DIR="${output_dir}"
  export RICH_TEST_POINT_COUNT=32768
  export RICH_TEST_FINAL_TIME=3.20604
  export RICH_TEST_INITIAL_BIN="${initial_bin}"
  export RICH_TEST_MAX_CYCLES=10000
  export RICH_TEST_DISABLE_AMR=1
  unset RICH_QUIET
  export RICH_TEST_PROGRESS_CYCLES="${RICH_TEST_PROGRESS_CYCLES:-100}"
  export RICH_TEST_PROGRESS_WALL_SECONDS="${RICH_TEST_PROGRESS_WALL_SECONDS:-60}"
  export OMP_NUM_THREADS=1
  unset RICH_INDIVIDUAL_MODE RICH_TEST_PRESCRIBED_AMR
  cd "${case_dir}"
  set +e
  mpirun -np 8 --bind-to core --map-by ppr:8:node "${rich_bin}" \
    > "${output_dir}/run.log" 2>&1
  local status=$?
  set -e
  awk '/MG_BICGSTAB_(PROGRESS|RELIABLE_UPDATE|CONVERGENCE|TIMING|TOLERANCE)|MG_PRECONDITIONER_(SETUP|APPLY)|MG_(SPECTRAL_POSITIVITY|PASSIVE_ROUNDOFF)_REPAIR|MG_TIMESTEP_LIMIT|MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH|Radiation time step ID|kp=|Group number|Total iterations:|Exited BiCGSTAB|Converged at iter|not good end|Negative raw radiation candidate energy/' \
    "${output_dir}/run.log" > "${output_dir}/mg_solver_diagnostics.log"
  printf '%s\n' "${status}" > "${output_dir}/exit_code.txt"
  return 0
}

run_calibration dt 10
run_calibration dt_half 9
/software/x86_64/5.14.0/python/3.12.1/bin/python3 \
  "${case_dir}/compare_calibration.py" \
  --dt-dir "${calibration_root}/dt" \
  --half-dir "${calibration_root}/dt_half" \
  --output "${calibration_root}/verdict.txt"
