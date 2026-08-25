#!/bin/bash

set -euo pipefail

case_dir="${RICH_FMM_ITS_CASE_DIR:?RICH_FMM_ITS_CASE_DIR is required}"
campaign_root="${RICH_FMM_ITS_CAMPAIGN_ROOT:?RICH_FMM_ITS_CAMPAIGN_ROOT is required}"
input_binary="${RICH_FMM_ITS_BINARY:?RICH_FMM_ITS_BINARY is required}"
expected_binary_sha="${RICH_FMM_ITS_BINARY_SHA256:?RICH_FMM_ITS_BINARY_SHA256 is required}"
expected_source_manifest_sha="${RICH_FMM_ITS_SOURCE_MANIFEST_SHA256:?RICH_FMM_ITS_SOURCE_MANIFEST_SHA256 is required}"
python_binary=/software/x86_64/5.14.0/python/3.12.1/bin/python3
python_library_directory=/software/x86_64/5.14.0/python/3.12.1/lib
python_ld_library_path="${python_library_directory}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
rank_count="${SLURM_NTASKS:-16}"
parity_side=20
performance_side=48
empty_side=2
lane_timeout_seconds=1200
process_group_cleanup_failure=92

case_dir="$(cd "${case_dir}" && pwd)"
repo_root="$(cd "${case_dir}/../.." && pwd)"
live_runner="$(readlink -f "${BASH_SOURCE[0]}")"

source_manifest_tmp=""
active_launcher_pid=""
active_launcher_pgid=""
finalized=0
campaign_root_created=0

drain_process_group() {
  local process_group="$1"
  local cleanup_log="$2"
  local attempt

  if [[ ! "${process_group}" =~ ^[0-9]+$ ]] || (( process_group <= 1 )); then
    printf 'invalid_process_group=%s\n' "${process_group}" >> "${cleanup_log}"
    return 2
  fi
  if ! kill -0 -- "-${process_group}" 2>/dev/null; then
    printf 'process_group=%s already_empty\n' "${process_group}" >> "${cleanup_log}"
    return 0
  fi

  printf 'process_group=%s signal=TERM\n' "${process_group}" >> "${cleanup_log}"
  kill -TERM -- "-${process_group}" 2>/dev/null || true
  for (( attempt = 0; attempt < 50; ++attempt )); do
    if ! kill -0 -- "-${process_group}" 2>/dev/null; then
      printf 'process_group=%s drained_after=TERM\n' "${process_group}" >> "${cleanup_log}"
      return 0
    fi
    sleep 0.1
  done

  printf 'process_group=%s signal=KILL\n' "${process_group}" >> "${cleanup_log}"
  kill -KILL -- "-${process_group}" 2>/dev/null || true
  for (( attempt = 0; attempt < 50; ++attempt )); do
    if ! kill -0 -- "-${process_group}" 2>/dev/null; then
      printf 'process_group=%s drained_after=KILL\n' "${process_group}" >> "${cleanup_log}"
      return 0
    fi
    sleep 0.1
  done

  printf 'process_group=%s cleanup_failed\n' "${process_group}" >> "${cleanup_log}"
  return 1
}

on_bootstrap_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  if [[ -n "${source_manifest_tmp}" ]]; then
    rm -f "${source_manifest_tmp}"
  fi
  if [[ "${campaign_root_created}" == 1 ]]; then
    printf '%s\n' "${status}" > "${campaign_root}/runner_exit_code.txt"
    printf '%s\n' "not_run" > "${campaign_root}/analyzer_exit_code.txt"
    printf '%s\n' "${status}" > "${campaign_root}/campaign_exit_code.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${campaign_root}/campaign_finished"
    printf 'bootstrap_failure\n' > "${campaign_root}/BOOTSTRAP_FAILURE"
  fi
  exit "${status}"
}

finalize_campaign() {
  local runner_status="$1"
  local analyzer_status=0
  local final_status="${runner_status}"
  if [[ "${finalized}" == 1 ]]; then
    return "${runner_status}"
  fi
  finalized=1
  date -u +%Y-%m-%dT%H:%M:%SZ > "${campaign_root}/campaign_lanes_finished"
  set +e
  env LD_LIBRARY_PATH="${python_ld_library_path}" \
    "${python_binary}" -B "${campaign_root}/frozen/analyze.py" --root "${campaign_root}" \
    > "${campaign_root}/analyzer_stdout.log" \
    2> "${campaign_root}/analyzer_stderr.log"
  analyzer_status=$?
  if [[ "${final_status}" == 0 && "${analyzer_status}" != 0 ]]; then
    final_status="${analyzer_status}"
  fi
  printf '%s\n' "${runner_status}" > "${campaign_root}/runner_exit_code.txt"
  printf '%s\n' "${analyzer_status}" > "${campaign_root}/analyzer_exit_code.txt"
  printf '%s\n' "${final_status}" > "${campaign_root}/campaign_exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${campaign_root}/campaign_finished"
  return "${final_status}"
}

on_exit() {
  local status=$?
  local final_status
  trap - EXIT HUP INT TERM
  if [[ -n "${source_manifest_tmp}" ]]; then
    rm -f "${source_manifest_tmp}"
  fi
  if [[ "${finalized}" == 0 ]]; then
    set +e
    finalize_campaign "${status}"
    final_status=$?
    set -e
    status="${final_status}"
  fi
  exit "${status}"
}

on_signal() {
  local signal_name="$1"
  local signal_status="$2"
  local child_status=0
  local cleanup_status=0
  trap - HUP INT TERM
  printf '%s\n' "${signal_name}" > "${campaign_root}/termination_signal.txt"
  if [[ -n "${active_launcher_pid}" ]]; then
    set +e
    kill -s "${signal_name}" -- "-${active_launcher_pgid}" 2>/dev/null || \
      kill -s "${signal_name}" "${active_launcher_pid}" 2>/dev/null
    wait "${active_launcher_pid}"
    child_status=$?
    drain_process_group "${active_launcher_pgid}" \
      "${campaign_root}/termination_cleanup.log"
    cleanup_status=$?
    set -e
    printf '%s\n' "${child_status}" > "${campaign_root}/terminated_launcher_exit_code.txt"
    printf '%s\n' "${cleanup_status}" > "${campaign_root}/termination_cleanup_exit_code.txt"
    active_launcher_pid=""
    active_launcher_pgid=""
  fi
  exit "${signal_status}"
}

if [[ "${rank_count}" != 16 ]]; then
  echo "Benchmark requires exactly 16 MPI ranks; got ${rank_count}" >&2
  exit 2
fi
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  echo "Benchmark must run inside a Slurm job allocation" >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-}" != 1 ]]; then
  echo "Benchmark requires exactly one allocated node" >&2
  exit 2
fi
if [[ ! -x "${input_binary}" ]]; then
  echo "Benchmark binary is missing or not executable: ${input_binary}" >&2
  exit 2
fi
if [[ ! -x "${python_binary}" ||
      ! -f "${python_library_directory}/libpython3.12.so.1.0" ]]; then
  echo "Pinned benchmark Python runtime is incomplete" >&2
  exit 2
fi
if ! python_version="$(env LD_LIBRARY_PATH="${python_ld_library_path}" \
       "${python_binary}" --version 2>&1)"; then
  echo "Pinned benchmark Python runtime does not start" >&2
  exit 2
fi
if ! env LD_LIBRARY_PATH="${python_ld_library_path}" \
       "${python_binary}" -B -c \
       'import argparse, hashlib, json, math, pathlib, re, statistics'; then
  echo "Pinned benchmark Python standard-library preflight failed" >&2
  exit 2
fi
python_dependencies="$(env LD_LIBRARY_PATH="${python_ld_library_path}" \
  ldd "${python_binary}")"
if [[ "${python_dependencies}" == *"not found"* ]]; then
  echo "Pinned benchmark Python has unresolved shared libraries" >&2
  exit 2
fi
input_binary="$(readlink -f "${input_binary}")"
actual_binary_sha="$(sha256sum "${input_binary}" | awk '{print $1}')"
if [[ "${actual_binary_sha}" != "${expected_binary_sha}" ]]; then
  echo "Benchmark binary SHA-256 changed before launch" >&2
  exit 2
fi
if [[ -e "${campaign_root}" ]]; then
  echo "Campaign root must not exist before launch: ${campaign_root}" >&2
  exit 2
fi

build_directory="$(dirname "${input_binary}")"
build_provenance="${build_directory}/FMM_INDIVIDUAL_BUILD_PROVENANCE.txt"
build_source_manifest="${build_directory}/FMM_INDIVIDUAL_SOURCE_MANIFEST.sha256"
build_environment="${build_directory}/FMM_INDIVIDUAL_BUILD_ENVIRONMENT.txt"
if [[ ! -s "${build_provenance}" || ! -s "${build_source_manifest}" ||
      ! -s "${build_environment}" ]]; then
  echo "Benchmark binary lacks build-time source provenance" >&2
  exit 2
fi
build_binary_sha="$(awk -F= '$1 == "binary_sha256" {print $2}' "${build_provenance}")"
build_source_manifest_sha="$(awk -F= '$1 == "source_manifest_sha256" {print $2}' "${build_provenance}")"
build_environment_sha="$(awk -F= '$1 == "build_environment_sha256" {print $2}' "${build_provenance}")"
actual_build_environment_sha="$(sha256sum "${build_environment}" | awk '{print $1}')"
actual_build_source_manifest_sha="$(sha256sum "${build_source_manifest}" | awk '{print $1}')"
if [[ "${build_binary_sha}" != "${actual_binary_sha}" ||
      "${build_source_manifest_sha}" != "${expected_source_manifest_sha}" ||
      "${actual_build_environment_sha}" != "${build_environment_sha}" ]]; then
  echo "Binary and requested source manifest do not match build provenance" >&2
  exit 2
fi
if [[ "${actual_build_source_manifest_sha}" != "${build_source_manifest_sha}" ]]; then
  echo "Build-time source manifest content changed" >&2
  exit 2
fi

source_manifest_tmp="$(mktemp)"
trap 'rm -f "${source_manifest_tmp}"' EXIT
bash "${case_dir}/source_manifest.sh" "${repo_root}" > "${source_manifest_tmp}"
actual_source_manifest_sha="$(sha256sum "${source_manifest_tmp}" | awk '{print $1}')"
if [[ "${actual_source_manifest_sha}" != "${expected_source_manifest_sha}" ]] ||
   ! cmp -s "${source_manifest_tmp}" "${build_source_manifest}"; then
  echo "Runtime source manifest differs from the source used for this binary" >&2
  echo "expected=${expected_source_manifest_sha}" >&2
  echo "actual=${actual_source_manifest_sha}" >&2
  exit 2
fi

trap on_bootstrap_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
if ! mkdir "${campaign_root}"; then
  echo "Campaign root creation was not atomic: ${campaign_root}" >&2
  exit 2
fi
campaign_root_created=1
mkdir "${campaign_root}/frozen" "${campaign_root}/tmp"
{
  echo "python_binary=${python_binary}"
  echo "python_library_directory=${python_library_directory}"
  echo "python_version=${python_version}"
  echo "python_preflight_exit=0"
  echo "dependencies_begin"
  printf '%s\n' "${python_dependencies}"
  echo "dependencies_end"
} > "${campaign_root}/PYTHON_RUNTIME.txt"
cp "${case_dir}/analyze.py" "${campaign_root}/frozen/analyze.py"
chmod 0555 "${campaign_root}/frozen/analyze.py"
trap on_exit EXIT
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

mv "${source_manifest_tmp}" "${campaign_root}/SOURCE_MANIFEST.sha256"
source_manifest_tmp=""
printf '1\n' > "${campaign_root}/ROOT_WAS_FRESH"
date -u +%Y-%m-%dT%H:%M:%SZ > "${campaign_root}/campaign_started"

frozen_binary="${campaign_root}/frozen/rich_fmm_individual_benchmark"
cp "${input_binary}" "${frozen_binary}"
for file in README.md build_benchmark.sh run_campaign.sh source_manifest.sh submit.sbatch test.cpp; do
  cp "${case_dir}/${file}" "${campaign_root}/frozen/${file}"
done
cp "${build_provenance}" "${campaign_root}/BUILD_PROVENANCE.txt"
cp "${build_source_manifest}" "${campaign_root}/BUILD_SOURCE_MANIFEST.sha256"
cp "${build_environment}" "${campaign_root}/BUILD_ENVIRONMENT.txt"
cp "${build_provenance}" "${campaign_root}/frozen/BUILD_PROVENANCE.txt"
cp "${build_source_manifest}" "${campaign_root}/frozen/BUILD_SOURCE_MANIFEST.sha256"
cp "${build_environment}" "${campaign_root}/frozen/BUILD_ENVIRONMENT.txt"
chmod 0555 "${frozen_binary}" \
  "${campaign_root}/frozen/build_benchmark.sh" \
  "${campaign_root}/frozen/run_campaign.sh" \
  "${campaign_root}/frozen/source_manifest.sh" \
  "${campaign_root}/frozen/submit.sbatch"
chmod 0444 "${campaign_root}/frozen/README.md" \
  "${campaign_root}/frozen/test.cpp" \
  "${campaign_root}/frozen/BUILD_PROVENANCE.txt" \
  "${campaign_root}/frozen/BUILD_SOURCE_MANIFEST.sha256" \
  "${campaign_root}/frozen/BUILD_ENVIRONMENT.txt"

source_manifest_tmp="$(mktemp)"
bash "${case_dir}/source_manifest.sh" "${repo_root}" > "${source_manifest_tmp}"
if ! cmp -s "${source_manifest_tmp}" "${build_source_manifest}"; then
  echo "Runtime source changed while the campaign bundle was frozen" >&2
  exit 2
fi
rm -f "${source_manifest_tmp}"
source_manifest_tmp=""
for file in README.md analyze.py build_benchmark.sh run_campaign.sh source_manifest.sh submit.sbatch test.cpp; do
  relative_path="runs/FmmIndividualTimestepBenchmark/${file}"
  expected_frozen_sha="$(awk -v path="${relative_path}" '$2 == path {print $1}' "${build_source_manifest}")"
  actual_frozen_sha="$(sha256sum "${campaign_root}/frozen/${file}" | awk '{print $1}')"
  if [[ -z "${expected_frozen_sha}" || "${actual_frozen_sha}" != "${expected_frozen_sha}" ]]; then
    echo "Frozen benchmark file does not match build manifest: ${file}" >&2
    exit 2
  fi
done
(
  cd "${campaign_root}"
  sha256sum \
    frozen/rich_fmm_individual_benchmark \
    frozen/analyze.py \
    frozen/README.md \
    frozen/build_benchmark.sh \
    frozen/run_campaign.sh \
    frozen/source_manifest.sh \
    frozen/submit.sbatch \
    frozen/test.cpp \
    frozen/BUILD_PROVENANCE.txt \
    frozen/BUILD_SOURCE_MANIFEST.sha256 \
    frozen/BUILD_ENVIRONMENT.txt > FROZEN_ARTIFACTS.sha256
)

live_runner_sha="$(sha256sum "${live_runner}" | awk '{print $1}')"
frozen_runner_sha="$(sha256sum "${campaign_root}/frozen/run_campaign.sh" | awk '{print $1}')"
{
  echo "root_was_fresh=1"
  echo "binary_input=${input_binary}"
  echo "binary_sha256=${actual_binary_sha}"
  echo "source_manifest_sha256=${actual_source_manifest_sha}"
  echo "build_provenance_sha256=$(sha256sum "${build_provenance}" | awk '{print $1}')"
  echo "source_head=$(git -C "${repo_root}" rev-parse HEAD)"
  echo "live_runner_sha256=${live_runner_sha}"
  echo "frozen_runner_sha256=${frozen_runner_sha}"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "rank_count=${rank_count}"
  echo "empty_side=${empty_side}"
  echo "parity_side=${parity_side}"
  echo "performance_side=${performance_side}"
  echo "lane_timeout_seconds=${lane_timeout_seconds}"
  echo "mpi_binding=core"
  echo "mpi_mapping=core"
  echo "hostlist=${SLURM_JOB_NODELIST:-local}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${campaign_root}/CAMPAIGN_PROVENANCE.txt"
git -C "${repo_root}" status --porcelain=v1 > "${campaign_root}/git_status.txt"
git -C "${repo_root}" diff HEAD --binary > "${campaign_root}/tracked_diff.patch"
git -C "${repo_root}" ls-files --others --exclude-standard > "${campaign_root}/untracked_files.txt"
git -C "${repo_root}" submodule status --recursive > "${campaign_root}/submodule_status.txt" || true
{
  command -v gcc
  gcc --version | head -n 1
  command -v mpirun
  mpirun --version | head -n 2
  command -v ompi_info
  ompi_info --version | head -n 2
  command -v setsid
  command -v timeout
} > "${campaign_root}/compiler_mpi_identity.txt" 2>&1
module -t list > "${campaign_root}/loaded_modules.txt" 2>&1 || true
ldd "${frozen_binary}" > "${campaign_root}/binary_ldd.txt"
uname -a > "${campaign_root}/uname.txt"
lscpu --extended=CPU,NODE,SOCKET,CORE,ONLINE > "${campaign_root}/cpu_topology.txt"
{
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-local}"
  echo "slurm_ntasks=${SLURM_NTASKS:-unset}"
  taskset -cp $$ 2>&1 || true
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -d "${SLURM_JOB_ID}" || true
  fi
} > "${campaign_root}/allocation_identity.txt" 2>&1
if [[ -f "$(dirname "${input_binary}")/CMakeCache.txt" ]]; then
  cp "$(dirname "${input_binary}")/CMakeCache.txt" "${campaign_root}/CMakeCache.txt"
fi

export OMP_NUM_THREADS=1
export RICH_FMM_TRACE=0
export RICH_INDIVIDUAL_PERF_TRACE=0
unset UCX_TLS
unset RICH_FMM_GEOM_LOG RICH_FMM_NONUNIFORM_DIAGNOSTICS RICH_FMM_PROGRESS_INTERVAL
unset RICH_INDIVIDUAL_ALL_ACTIVE_CELL_UPDATE
unset RICH_INDIVIDUAL_AUTO_REBALANCE
unset RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH
unset RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN
unset RICH_INDIVIDUAL_IDENTITY_OWNED_MOVE
unset RICH_INDIVIDUAL_REBALANCE_AMORTIZATION
unset RICH_INDIVIDUAL_REBALANCE_AMR_THRESHOLD
unset RICH_INDIVIDUAL_REBALANCE_BEFORE_FIRST_EVENT
unset RICH_INDIVIDUAL_REBALANCE_COOLDOWN
unset RICH_INDIVIDUAL_REBALANCE_THRESHOLD
unset RICH_INDIVIDUAL_REUSE_EVENT_MESH
unset RICH_INDIVIDUAL_REUSE_EVENT_MESH_SHADOW

verify_frozen_artifacts() {
  local lane="$1"
  local check_log="${campaign_root}/${lane}/frozen_rehash.txt"
  local current_binary_sha
  local current_live_runner_sha
  local current_frozen_runner_sha
  current_binary_sha="$(sha256sum "${frozen_binary}" | awk '{print $1}')"
  current_live_runner_sha="$(sha256sum "${live_runner}" | awk '{print $1}')"
  current_frozen_runner_sha="$(sha256sum "${campaign_root}/frozen/run_campaign.sh" | awk '{print $1}')"
  {
    echo "binary_sha256=${current_binary_sha}"
    echo "live_runner_sha256=${current_live_runner_sha}"
    echo "frozen_runner_sha256=${current_frozen_runner_sha}"
    if [[ "${current_binary_sha}" != "${expected_binary_sha}" ]]; then
      echo "Frozen binary SHA-256 changed before lane ${lane}"
      return 90
    fi
    if [[ "${current_live_runner_sha}" != "${current_frozen_runner_sha}" ]]; then
      echo "Live campaign runner differs from the frozen runner before lane ${lane}"
      return 91
    fi
    (
      cd "${campaign_root}"
      sha256sum --check FROZEN_ARTIFACTS.sha256
    )
  } > "${check_log}" 2>&1
}

run_lane() {
  local lane="$1"
  local scenario="$2"
  local mode="$3"
  local grid_side="$4"
  local final_tick="$5"
  local lane_rank_count="${6:-${rank_count}}"
  local lane_dir="${campaign_root}/${lane}"
  local lane_status=0
  local cleanup_status=0
  mkdir -p "${lane_dir}" "${campaign_root}/tmp/${lane}"
  set +e
  verify_frozen_artifacts "${lane}"
  lane_status=$?
  set -e
  if [[ "${lane_status}" != 0 ]]; then
    printf '%s\n' "${lane_status}" > "${lane_dir}/exit_code.txt"
    return "${lane_status}"
  fi
  export RICH_FMM_ITS_SCENARIO="${scenario}"
  export RICH_FMM_ITS_MODE="${mode}"
  export RICH_FMM_ITS_GRID_SIDE="${grid_side}"
  export RICH_FMM_ITS_FINAL_TICK="${final_tick}"
  export RICH_FMM_ITS_OUTPUT="${lane_dir}"
  export TMPDIR="${campaign_root}/tmp/${lane}"
  export TMP="${TMPDIR}"
  export TEMP="${TMPDIR}"
  export OMPI_MCA_orte_tmpdir_base="${TMPDIR}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${lane_dir}/started_utc.txt"
  setsid timeout --signal=TERM --kill-after=60s \
    "${lane_timeout_seconds}s" \
    mpirun --bind-to core --map-by core --report-bindings \
      -np "${lane_rank_count}" "${frozen_binary}" \
      > "${lane_dir}/stdout.log" 2> "${lane_dir}/stderr.log" &
  active_launcher_pid=$!
  active_launcher_pgid="${active_launcher_pid}"
  {
    echo "pid=${active_launcher_pid}"
    echo "pgid=${active_launcher_pgid}"
    echo "timeout_seconds=${lane_timeout_seconds}"
    echo "rank_count=${lane_rank_count}"
    echo "binding=core"
    echo "mapping=core"
  } > "${lane_dir}/launcher_identity.txt"
  set +e
  wait "${active_launcher_pid}"
  lane_status=$?
  drain_process_group "${active_launcher_pgid}" "${lane_dir}/launcher_cleanup.log"
  cleanup_status=$?
  set -e
  printf '%s\n' "${cleanup_status}" > "${lane_dir}/launcher_cleanup_exit_code.txt"
  if [[ "${cleanup_status}" != 0 ]]; then
    lane_status="${process_group_cleanup_failure}"
  fi
  active_launcher_pid=""
  active_launcher_pgid=""
  printf '%s\n' "${lane_status}" > "${lane_dir}/exit_code.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${lane_dir}/finished_utc.txt"
  return "${lane_status}"
}

runner_status=0
run_and_record() {
  local lane_status=0
  if run_lane "$@"; then
    lane_status=0
  else
    lane_status=$?
  fi
  if [[ "${runner_status}" == 0 && "${lane_status}" != 0 ]]; then
    runner_status="${lane_status}"
  fi
  if [[ "${lane_status}" == "${process_group_cleanup_failure}" ]]; then
    return "${process_group_cleanup_failure}"
  fi
  return 0
}

run_and_record primitive_recovery_probe primitive_recovery_probe individual_sync 2 1 1
run_and_record warmup_global performance global "${performance_side}" 4
run_and_record warmup_individual performance individual_sparse "${performance_side}" 4
run_and_record empty_global parity global "${empty_side}" 16
run_and_record empty_individual parity individual_sparse "${empty_side}" 16
run_and_record parity_global parity global "${parity_side}" 16
run_and_record parity_individual parity individual_sync "${parity_side}" 16
run_and_record perf_global_1 performance global "${performance_side}" 64
run_and_record perf_individual_1 performance individual_sparse "${performance_side}" 64
run_and_record perf_individual_2 performance individual_sparse "${performance_side}" 64
run_and_record perf_global_2 performance global "${performance_side}" 64
run_and_record perf_global_3 performance global "${performance_side}" 64
run_and_record perf_individual_3 performance individual_sparse "${performance_side}" 64

set +e
finalize_campaign "${runner_status}"
final_status=$?
set -e
trap - EXIT HUP INT TERM
exit "${final_status}"
