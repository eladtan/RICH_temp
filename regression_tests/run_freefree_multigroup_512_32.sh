#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGRESSION_DIR="${ROOT_DIR}/regression_tests"
BUILD_SCRIPT="${ROOT_DIR}/build_rich.sh"
COMPARE_PLOTTER="${REGRESSION_DIR}/lib/plot_eulerian_diffusion_freefree_multigroup_compare.py"

CONFIG="${1:-intelReleaseMPI}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SLURM_NTASKS="${SLURM_NTASKS:-8}"
SLURM_PARTITION="${SLURM_PARTITION:-bigrun}"
SLURM_EXCLUSIVE="${SLURM_EXCLUSIVE:-1}"
ENERGY_GROUPS_NUM="${ENERGY_GROUPS_NUM:-32}"
RUN_LOCAL="${RUN_LOCAL:-0}"

if [[ ! -x "${BUILD_SCRIPT}" ]]; then
    echo "Missing executable build helper: ${BUILD_SCRIPT}" >&2
    exit 2
fi

if [[ ! -f "${COMPARE_PLOTTER}" ]]; then
    echo "Missing compare plotter: ${COMPARE_PLOTTER}" >&2
    exit 2
fi

if ! command -v ml >/dev/null 2>&1; then
    echo "The 'ml' command is required to load OpenMPI before MPI runs" >&2
    exit 2
fi

if [[ "${RUN_LOCAL}" -ne 1 ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "The 'sbatch' command is required for multigroup free-free MPI regression runs (use --local to bypass)" >&2
    exit 2
fi

ml openmpi/4.1.6/Intel/OneApi/2024.2.1

declare -a CASES=(
    "eulerian_diffusion_freefree_multigroup_1d"
    "eulerian_diffusion_freefree_multigroup_1d_512_limited"
    "eulerian_diffusion_freefree_multigroup_1d_32"
    "eulerian_diffusion_freefree_multigroup_1d_32_limited"
)

is_nonempty_and_newer() {
    local file_path="$1"
    local start_epoch="$2"
    local file_epoch
    local attempt

    for attempt in 1 2 3; do
        ls "$(dirname "$file_path")" > /dev/null 2>&1 || true
        if [[ -s "$file_path" ]]; then
            file_epoch="$(stat -c %Y "$file_path" 2>/dev/null || true)"
            if [[ -n "$file_epoch" && "$file_epoch" -ge "$start_epoch" ]]; then
                return 0
            fi
        fi
        [[ "$attempt" -lt 3 ]] && sleep 2
    done
    return 1
}

run_case() {
    local case_id="$1"
    local case_rel="regression_tests/cases/${case_id}"
    local case_dir="${ROOT_DIR}/${case_rel}"
    local build_stdout="${case_dir}/build.stdout.log"
    local build_stderr="${case_dir}/build.stderr.log"
    local run_stdout="${case_dir}/run.stdout.log"
    local run_stderr="${case_dir}/run.stderr.log"
    local rich_bin="${ROOT_DIR}/build/${CONFIG}/${case_id}/rich"
    local run_start_epoch
    local case_ntasks="${SLURM_NTASKS}"

    if [[ ! -d "${case_dir}" ]]; then
        echo "Missing regression case directory: ${case_dir}" >&2
        exit 2
    fi

    echo "[BUILD] ${case_id}"
    (
        cd "${ROOT_DIR}"
        VERBOSE=1 "${BUILD_SCRIPT}" "${CONFIG}" \
            "--test_name=${case_rel}" \
            "--build-subdir=${case_id}" \
            "--energy_groups_num=${ENERGY_GROUPS_NUM}"
    ) >"${build_stdout}" 2>"${build_stderr}"

    if [[ ! -x "${rich_bin}" && ! -L "${rich_bin}" ]]; then
        echo "Expected built binary missing: ${rich_bin}" >&2
        exit 1
    fi

    echo "[RUN] ${case_id}"
    if [[ "${case_id}" == *"_32"* ]]; then
        case_ntasks=4
    fi
    # Avoid stale artifacts from previous runs causing false positives.
    rm -f "${case_dir}/temperature_profile.txt" "${case_dir}/shock_position.txt"
    rm -f "${run_stdout}" "${run_stderr}"
    : > "${run_stdout}"
    : > "${run_stderr}"
    run_start_epoch="$(date +%s)"

    run_rc=0
    if [[ "${RUN_LOCAL}" -eq 1 ]]; then
        (cd "${case_dir}" && mpirun -np "${case_ntasks}" "${rich_bin}") \
            >"${run_stdout}" 2>"${run_stderr}" || run_rc=$?

        if [[ "${run_rc}" -ne 0 ]]; then
            echo "Local mpirun failed for ${case_id} (exit code ${run_rc})" >&2
            exit "${run_rc}"
        fi
    else
        sbatch_args=(
            sbatch
            --wait
            --job-name="${case_id}"
            --ntasks="${case_ntasks}"
            --partition="${SLURM_PARTITION}"
            --output="${run_stdout}"
            --error="${run_stderr}"
            --chdir="${case_dir}"
            --wrap "mpirun -np ${case_ntasks} \"${rich_bin}\""
        )
        if [[ "${SLURM_EXCLUSIVE}" == "1" ]]; then
            sbatch_args+=(--exclusive)
        fi

        # Stream full case stdout/stderr into this suite log in real time.
        tail -n +1 -f "${run_stdout}" &
        tail_stdout_pid=$!
        tail -n +1 -f "${run_stderr}" >&2 &
        tail_stderr_pid=$!

        submit_line=""
        submit_line="$("${sbatch_args[@]}" 2>>"${run_stderr}")" || run_rc=$?
        submit_line="$(tr -d '\r' <<< "${submit_line}")"
        [[ -n "${submit_line}" ]] && echo "${submit_line}"
        job_id="$(awk '/Submitted batch job/ {print $4}' <<< "${submit_line}")"
        if [[ -z "${job_id}" ]]; then
            job_id="unknown"
        fi
        kill "${tail_stdout_pid}" "${tail_stderr_pid}" >/dev/null 2>&1 || true
        wait "${tail_stdout_pid}" "${tail_stderr_pid}" >/dev/null 2>&1 || true

        if [[ "${run_rc}" -ne 0 ]]; then
            echo "SLURM run failed for ${case_id} (job ${job_id}, exit code ${run_rc})" >&2
            exit "${run_rc}"
        fi
    fi

    if ! is_nonempty_and_newer "${case_dir}/temperature_profile.txt" "${run_start_epoch}"; then
        echo "Missing profile output for ${case_id}: ${case_dir}/temperature_profile.txt" >&2
        exit 1
    fi
    if ! is_nonempty_and_newer "${case_dir}/shock_position.txt" "${run_start_epoch}"; then
        echo "Missing shock output for ${case_id}: ${case_dir}/shock_position.txt" >&2
        exit 1
    fi
}

for case_id in "${CASES[@]}"; do
    run_case "${case_id}"
done

PROFILE_512="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_multigroup_1d/temperature_profile.txt"
PROFILE_512_LIMITED="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_multigroup_1d_512_limited/temperature_profile.txt"
PROFILE_32="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_multigroup_1d_32/temperature_profile.txt"
PROFILE_32_LIMITED="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_multigroup_1d_32_limited/temperature_profile.txt"
OUTPUT_DIR="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_multigroup_compare"

if [[ ! -s "${PROFILE_512}" ]]; then
    echo "Missing profile output for 512-cell case: ${PROFILE_512}" >&2
    exit 1
fi

if [[ ! -s "${PROFILE_512_LIMITED}" ]]; then
    echo "Missing profile output for 512-cell limited case: ${PROFILE_512_LIMITED}" >&2
    exit 1
fi

if [[ ! -s "${PROFILE_32}" ]]; then
    echo "Missing profile output for 32-cell case: ${PROFILE_32}" >&2
    exit 1
fi

if [[ ! -s "${PROFILE_32_LIMITED}" ]]; then
    echo "Missing profile output for 32-cell limited case: ${PROFILE_32_LIMITED}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${COMPARE_PLOTTER}" \
    --profile-512 "${PROFILE_512}" \
    --profile-512-limited "${PROFILE_512_LIMITED}" \
    --profile-32 "${PROFILE_32}" \
    --profile-32-limited "${PROFILE_32_LIMITED}" \
    --output-dir "${OUTPUT_DIR}"

echo "Multigroup comparison plots generated in: ${OUTPUT_DIR}"
