#!/bin/bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s global|full-variable|partial\n' "$0" >&2
  exit 2
fi
lane="$1"
case "${lane}" in
  global|full-variable|partial) ;;
  *) printf 'invalid lane: %s\n' "${lane}" >&2; exit 2 ;;
esac

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../../.." && pwd)"
campaign_root="${RICH_CAMPAIGN_ROOT:?RICH_CAMPAIGN_ROOT must name the shared campaign directory}"
lane_dir="${campaign_root}/${lane}"
rich_bin="${RICH_CAMPAIGN_BINARY:-${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI}"
mkdir -p "${lane_dir}"

if [[ ! -x "${rich_bin}" ]]; then
  printf 'missing executable: %s\n' "${rich_bin}" > "${lane_dir}/driver_error.txt"
  printf '127\n' > "${lane_dir}/exit_code.txt"
  exit 127
fi

unset RICH_TEST_RESTART_INPUT RICH_TEST_ACCUMULATE_RESTART
restart_pointer="${lane_dir}/restart_checkpoint_latest.txt"
restart_input=""
if [[ -f "${restart_pointer}" ]]; then
  restart_input="$(awk '$1 == "snapshot" {print $2; exit}' "${restart_pointer}")"
  case "${restart_input}" in
    "${lane_dir}/restart_checkpoint_0.h5"|"${lane_dir}/restart_checkpoint_1.h5") ;;
    *)
      printf 'unsafe restart snapshot in %s: %s\n' \
        "${restart_pointer}" "${restart_input}" > "${lane_dir}/driver_error.txt"
      printf '76\n' > "${lane_dir}/exit_code.txt"
      exit 76
      ;;
  esac
  if [[ ! -s "${restart_input}" ]]; then
    printf 'missing restart snapshot: %s\n' "${restart_input}" \
      > "${lane_dir}/driver_error.txt"
    printf '76\n' > "${lane_dir}/exit_code.txt"
    exit 76
  fi
  for ((rank = 0; rank < ${SLURM_NTASKS:-128}; ++rank)); do
    if [[ ! -s "${restart_input}.benchmark_state_rank_${rank}.txt" ]]; then
      printf 'missing benchmark restart state for rank %d: %s\n' \
        "${rank}" "${restart_input}" > "${lane_dir}/driver_error.txt"
      printf '76\n' > "${lane_dir}/exit_code.txt"
      exit 76
    fi
  done
  export RICH_TEST_RESTART_INPUT="${restart_input}"
  export RICH_TEST_ACCUMULATE_RESTART=1
fi

export RICH_TEST_MG_PRECONDITIONER="${RICH_TEST_MG_PRECONDITIONER:-cell_block}"
unset RICH_MG_INDIVIDUAL_SHADOW_RESERVOIRS
if [[ "${lane}" == partial ]]; then
  export RICH_MG_INDIVIDUAL_PASSIVE_POLICY=dirichlet
else
  unset RICH_MG_INDIVIDUAL_PASSIVE_POLICY
fi

# A checkpoint records the monotonic forced-active latch, but a pre-latch
# checkpoint still needs the same threshold policy after restart. Recover the
# last production-segment policy when the replacement environment omits it,
# and reject an explicit mid-campaign policy change.
closure_min_bin="${RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN:-}"
closure_latch="${RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH:-}"
if [[ -n "${restart_input}" && -s "${lane_dir}/run_info.txt" ]]; then
  previous_closure_min_bin="$(awk '
    $1 == "individual_force_all_active_min_bin" { value = $2 }
    END { print value }
  ' "${lane_dir}/run_info.txt")"
  previous_closure_latch="$(awk '
    $1 == "individual_force_all_active_latch" { value = $2 }
    END { print value }
  ' "${lane_dir}/run_info.txt")"
  # Older segments recorded the threshold but predate the latch selector.
  if [[ -n "${previous_closure_min_bin}" && -z "${previous_closure_latch}" ]]; then
    previous_closure_latch=0
  fi
  if [[ -z "${previous_closure_min_bin}" && -n "${previous_closure_latch}" ]]; then
    printf 'incomplete forced-active policy in %s/run_info.txt\n' "${lane_dir}" \
      > "${lane_dir}/driver_error.txt"
    printf '76\n' > "${lane_dir}/exit_code.txt"
    exit 76
  fi
  if [[ -n "${previous_closure_min_bin}" ]]; then
    if [[ -z "${closure_min_bin}" && "${previous_closure_min_bin}" != disabled ]]; then
      closure_min_bin="${previous_closure_min_bin}"
    elif [[ "${closure_min_bin:-disabled}" != "${previous_closure_min_bin}" ]]; then
      printf 'forced-active minimum-bin policy changed across restart: %s -> %s\n' \
        "${previous_closure_min_bin}" "${closure_min_bin:-disabled}" \
        > "${lane_dir}/driver_error.txt"
      printf '76\n' > "${lane_dir}/exit_code.txt"
      exit 76
    fi
    if [[ -z "${closure_latch}" ]]; then
      closure_latch="${previous_closure_latch}"
    elif [[ "${closure_latch}" != "${previous_closure_latch}" ]]; then
      printf 'forced-active latch policy changed across restart: %s -> %s\n' \
        "${previous_closure_latch}" "${closure_latch}" \
        > "${lane_dir}/driver_error.txt"
      printf '76\n' > "${lane_dir}/exit_code.txt"
      exit 76
    fi
  fi
fi
closure_latch="${closure_latch:-0}"
if [[ "${closure_latch}" != 0 && "${closure_latch}" != 1 ]]; then
  printf 'invalid RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH: %s\n' \
    "${closure_latch}" > "${lane_dir}/driver_error.txt"
  printf '76\n' > "${lane_dir}/exit_code.txt"
  exit 76
fi
if [[ -n "${closure_min_bin}" ]]; then
  if [[ ! "${closure_min_bin}" =~ ^([1-9]|[1-5][0-9]|6[0-3])$ ]]; then
    printf 'invalid RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN: %s\n' \
      "${closure_min_bin}" > "${lane_dir}/driver_error.txt"
    printf '76\n' > "${lane_dir}/exit_code.txt"
    exit 76
  fi
  export RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN="${closure_min_bin}"
else
  unset RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN
fi
if [[ "${closure_latch}" == 1 && -z "${closure_min_bin}" ]]; then
  printf '%s\n' \
    'RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH=1 requires RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN' \
    > "${lane_dir}/driver_error.txt"
  printf '76\n' > "${lane_dir}/exit_code.txt"
  exit 76
fi
export RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH="${closure_latch}"

fail_active_hilbert_policy()
{
  printf '%s\n' "$1" > "${lane_dir}/driver_error.txt"
  printf '76\n' > "${lane_dir}/exit_code.txt"
  exit 76
}

normalize_active_hilbert_toggle()
{
  if [[ "$1" != 0 && "$1" != 1 ]]; then
    return 1
  fi
  printf '%s\n' "$1"
}

normalize_active_hilbert_skew()
{
  awk -v value="$1" 'BEGIN {
    if (value !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/)
      exit 1
    number = value + 0
    if (number < 1 || number > 1.7976931348623157e308)
      exit 1
    printf "%.17g\n", number
  }'
}

if [[ -n "${restart_input}" && ! -s "${lane_dir}/run_info.txt" ]]; then
  fail_active_hilbert_policy \
    "restart is missing ${lane_dir}/run_info.txt"
fi

active_hilbert_cache="${RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE:-}"
active_hilbert_threshold="${RICH_INDIVIDUAL_ACTIVE_HILBERT_THRESHOLD:-}"
active_hilbert_max_owned_skew="${RICH_INDIVIDUAL_ACTIVE_HILBERT_MAX_OWNED_SKEW:-}"
if [[ -n "${restart_input}" && -s "${lane_dir}/run_info.txt" ]]; then
  previous_active_hilbert_cache="$(awk '
    $1 == "individual_active_hilbert_cache" { value = $2 }
    END { print value }
  ' "${lane_dir}/run_info.txt")"
  previous_active_hilbert_threshold="$(awk '
    $1 == "individual_active_hilbert_threshold" { value = $2 }
    END { print value }
  ' "${lane_dir}/run_info.txt")"
  previous_active_hilbert_max_owned_skew="$(awk '
    $1 == "individual_active_hilbert_max_owned_skew" { value = $2 }
    END { print value }
  ' "${lane_dir}/run_info.txt")"

  if [[ -z "${previous_active_hilbert_cache}" &&
        -z "${previous_active_hilbert_threshold}" &&
        -z "${previous_active_hilbert_max_owned_skew}" ]]; then
    previous_active_hilbert_cache=0
    previous_active_hilbert_threshold=1.25
    previous_active_hilbert_max_owned_skew=2.0
  elif [[ -z "${previous_active_hilbert_cache}" ||
          -z "${previous_active_hilbert_threshold}" ||
          -z "${previous_active_hilbert_max_owned_skew}" ]]; then
    fail_active_hilbert_policy \
      "incomplete active Hilbert policy in ${lane_dir}/run_info.txt"
  fi

  if ! previous_active_hilbert_cache="$(
    normalize_active_hilbert_toggle "${previous_active_hilbert_cache}"
  )" ||
     ! previous_active_hilbert_threshold="$(
    normalize_active_hilbert_skew "${previous_active_hilbert_threshold}"
  )" ||
     ! previous_active_hilbert_max_owned_skew="$(
    normalize_active_hilbert_skew "${previous_active_hilbert_max_owned_skew}"
  )"; then
    fail_active_hilbert_policy \
      "invalid active Hilbert policy in ${lane_dir}/run_info.txt"
  fi

  active_hilbert_cache="${active_hilbert_cache:-${previous_active_hilbert_cache}}"
  active_hilbert_threshold="${active_hilbert_threshold:-${previous_active_hilbert_threshold}}"
  active_hilbert_max_owned_skew="${active_hilbert_max_owned_skew:-${previous_active_hilbert_max_owned_skew}}"
fi

active_hilbert_cache="${active_hilbert_cache:-1}"
active_hilbert_threshold="${active_hilbert_threshold:-1.25}"
active_hilbert_max_owned_skew="${active_hilbert_max_owned_skew:-2.0}"
if ! active_hilbert_cache="$(
  normalize_active_hilbert_toggle "${active_hilbert_cache}"
)" ||
   ! active_hilbert_threshold="$(
  normalize_active_hilbert_skew "${active_hilbert_threshold}"
)" ||
   ! active_hilbert_max_owned_skew="$(
  normalize_active_hilbert_skew "${active_hilbert_max_owned_skew}"
)"; then
  fail_active_hilbert_policy "invalid active Hilbert runtime policy"
fi

if [[ -n "${restart_input}" && -s "${lane_dir}/run_info.txt" ]] &&
   [[ "${active_hilbert_cache}" != "${previous_active_hilbert_cache}" ||
      "${active_hilbert_threshold}" != "${previous_active_hilbert_threshold}" ||
      "${active_hilbert_max_owned_skew}" != "${previous_active_hilbert_max_owned_skew}" ]]; then
  fail_active_hilbert_policy \
    "active Hilbert policy changed across restart"
fi

export RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE="${active_hilbert_cache}"
export RICH_INDIVIDUAL_ACTIVE_HILBERT_THRESHOLD="${active_hilbert_threshold}"
export RICH_INDIVIDUAL_ACTIVE_HILBERT_MAX_OWNED_SKEW="${active_hilbert_max_owned_skew}"

mpi_ranks_per_node="${RICH_MPI_RANKS_PER_NODE:-16}"
if [[ ! "${mpi_ranks_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'invalid RICH_MPI_RANKS_PER_NODE: %s\n' "${mpi_ranks_per_node}" \
    > "${lane_dir}/driver_error.txt"
  printf '76\n' > "${lane_dir}/exit_code.txt"
  exit 76
fi

{
  printf '%s\n' '--- production segment ---'
  printf 'lane %s\n' "${lane}"
  printf 'job_id %s\n' "${SLURM_JOB_ID:-none}"
  printf 'restart_count %s\n' "${SLURM_RESTART_COUNT:-0}"
  printf 'restart_input %s\n' "${restart_input:-none}"
  printf 'nodes %s\n' "${SLURM_JOB_NUM_NODES:-unknown}"
  printf 'tasks %s\n' "${SLURM_NTASKS:-128}"
  printf 'mpi_ranks_per_node %s\n' "${mpi_ranks_per_node}"
  printf 'mg_preconditioner %s\n' "${RICH_TEST_MG_PRECONDITIONER}"
  printf 'mg_individual_passive_policy %s\n' \
    "${RICH_MG_INDIVIDUAL_PASSIVE_POLICY:-library-default}"
  printf 'individual_force_all_active_min_bin %s\n' \
    "${RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN:-disabled}"
  printf 'individual_force_all_active_latch %s\n' \
    "${RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH}"
  printf 'individual_active_hilbert_cache %s\n' \
    "${RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE:-1}"
  printf 'individual_active_hilbert_threshold %s\n' \
    "${RICH_INDIVIDUAL_ACTIVE_HILBERT_THRESHOLD:-1.25}"
  printf 'individual_active_hilbert_max_owned_skew %s\n' \
    "${RICH_INDIVIDUAL_ACTIVE_HILBERT_MAX_OWNED_SKEW:-2.0}"
  printf 'individual_perf_trace %s\n' \
    "${RICH_INDIVIDUAL_PERF_TRACE:-0}"
  printf 'git_revision '
  git -C "${repo_root}" rev-parse HEAD
  sha256sum "${rich_bin}"
  printf 'mpi_launcher '
  command -v mpirun
  mpirun --version | head -n 1
  printf 'mpi_library '
  ldd "${rich_bin}" | awk '/libmpi\.so/{print $3; exit}'
  module list 2>&1
} >> "${lane_dir}/run_info.txt"

export THUNDER_ARTIFACT_DIR="${lane_dir}"
export RICH_TEST_POINT_COUNT="${RICH_TEST_POINT_COUNT:-2000000}"
export RICH_TEST_INITIAL_BIN="${RICH_TEST_INITIAL_BIN:-30}"
export RICH_TEST_MAXIMUM_BIN="${RICH_TEST_MAXIMUM_BIN:-40}"
export RICH_TEST_TIME_QUANTUM="${RICH_TEST_TIME_QUANTUM:-2.8475356730643257e-13}"
export RICH_TEST_MAX_GLOBAL_DT="${RICH_TEST_MAX_GLOBAL_DT:-$(
  awk -v quantum="${RICH_TEST_TIME_QUANTUM}" \
      -v bin="${RICH_TEST_MAXIMUM_BIN}" \
      'BEGIN { printf "%.17g", quantum * (2 ^ bin) }'
)}"
export RICH_TEST_MAX_CELLS="${RICH_TEST_MAX_CELLS:-3000000}"
export RICH_TEST_AMR_BATCH="${RICH_TEST_AMR_BATCH:-50000}"
unset RICH_QUIET
export RICH_TEST_PROGRESS_CYCLES="${RICH_TEST_PROGRESS_CYCLES:-100}"
export RICH_TEST_PROGRESS_WALL_SECONDS="${RICH_TEST_PROGRESS_WALL_SECONDS:-60}"
export RICH_TEST_CHECKPOINT_WALL_SECONDS="${RICH_TEST_CHECKPOINT_WALL_SECONDS:-21600}"
export RICH_TEST_SEGMENT_WALL_SECONDS="${RICH_TEST_SEGMENT_WALL_SECONDS:-1728000}"
export OMP_NUM_THREADS=1
unset RICH_TEST_DISABLE_AMR RICH_TEST_PRESCRIBED_AMR \
  RICH_TEST_MAX_CYCLES RICH_TEST_SPARSE_INITIAL_BIN \
  RICH_TEST_SPARSE_SINGLE_RANK RICH_TEST_FORCE_CENTRAL_POINT \
  RICH_TEST_REQUIRE_EMPTY_OWNED_RANK RICH_TEST_DISABLE_GRAVITY \
  RICH_TEST_DISABLE_COMPTON RICH_TEST_DISABLE_DOPPLER
if [[ "${lane}" == global ]]; then
  unset RICH_INDIVIDUAL_MODE
  unset RICH_MG_DISTRIBUTED_ACTIVE_PROFILE
else
  export RICH_INDIVIDUAL_MODE="${lane}"
  export RICH_MG_DISTRIBUTED_ACTIVE_PROFILE=1
fi

cd "${case_dir}"
if [[ -z "${restart_input}" ]]; then
  : > "${lane_dir}/run.log"
fi
{
  printf 'PRODUCTION_SEGMENT_START job_id=%s restart_count=%s restart_input=%s\n' \
    "${SLURM_JOB_ID:-none}" "${SLURM_RESTART_COUNT:-0}" \
    "${restart_input:-none}"
  printf 'PRODUCTION_RADIATION_POLICY lane=%s passive_policy=%s\n' \
    "${lane}" "${RICH_MG_INDIVIDUAL_PASSIVE_POLICY:-library-default}"
  printf 'PRODUCTION_FORCED_ACTIVE_POLICY lane=%s min_bin=%s latch=%s restart_input=%s\n' \
    "${lane}" "${RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN:-disabled}" \
    "${RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH}" \
    "${restart_input:-none}"
  printf 'PRODUCTION_ACTIVE_HILBERT_POLICY lane=%s enabled=%s threshold=%s max_owned_skew=%s\n' \
    "${lane}" "${RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE:-1}" \
    "${RICH_INDIVIDUAL_ACTIVE_HILBERT_THRESHOLD:-1.25}" \
    "${RICH_INDIVIDUAL_ACTIVE_HILBERT_MAX_OWNED_SKEW:-2.0}"
} >> "${lane_dir}/run.log"
set +e
mpirun -np "${SLURM_NTASKS:-128}" --bind-to core \
  --map-by "ppr:${mpi_ranks_per_node}:node" \
  "${rich_bin}" >> "${lane_dir}/run.log" 2>&1
status=$?
set -e
printf 'PRODUCTION_SEGMENT_END job_id=%s restart_count=%s status=%s\n' \
  "${SLURM_JOB_ID:-none}" "${SLURM_RESTART_COUNT:-0}" "${status}" \
  >> "${lane_dir}/run.log"
if [[ "${status}" -eq 75 ]]; then
  if [[ ! -s "${restart_pointer}" ]]; then
    printf 'segment exit 75 without a restart pointer\n' \
      > "${lane_dir}/driver_error.txt"
    printf '76\n' > "${lane_dir}/exit_code.txt"
    exit 76
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exit 75
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
awk '/MG_BICGSTAB_(PROGRESS|RELIABLE_UPDATE|CONVERGENCE|TIMING|TOLERANCE|HISTORICAL_POLICY)|MG_PRECONDITIONER_(SETUP|APPLY)|MG_(SPECTRAL_POSITIVITY|PASSIVE_ROUNDOFF)_REPAIR|MG_COMPTON_|MG_RADIATION_FORCE_|MG_ENERGY_BALANCE_REJECTION|MG_TIMESTEP_LIMIT|MG_INDIVIDUAL_(ALL_ACTIVE_FAST_PATH|PASSIVE_POLICY)|MG_ACTIVE_OWNERSHIP|INDIVIDUAL_(RADIATION_REJECTION|RADIATION_DEFECT|TERMINAL_EVENT_CLAMP)|Individual radiation retries:|Radiation time step ID|kp=|Group number|Total iterations:|Exited BiCGSTAB|Converged at iter|not good end|Negative raw radiation candidate energy/' \
  "${lane_dir}/run.log" > "${lane_dir}/mg_solver_diagnostics.log"
printf '%s\n' "${status}" > "${lane_dir}/exit_code.txt"
exit "${status}"
