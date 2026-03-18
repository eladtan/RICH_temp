#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGRESSION_DIR="${ROOT_DIR}/regression_tests"
BUILD_SCRIPT="${ROOT_DIR}/build_rich.sh"
COMPARE_PLOTTER="${REGRESSION_DIR}/lib/plot_eulerian_diffusion_freefree_compare.py"

CONFIG="${1:-intelReleaseMPI}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SLURM_NTASKS_512="${SLURM_NTASKS_512:-16}"
SLURM_NTASKS_32="${SLURM_NTASKS_32:-4}"
SLURM_PARTITION="${SLURM_PARTITION:-bigrun}"
SLURM_EXCLUSIVE="${SLURM_EXCLUSIVE:-1}"
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
    echo "The 'sbatch' command is required for free-free MPI regression runs (use --local to bypass)" >&2
    exit 2
fi

ml openmpi/4.1.6/Intel/OneApi/2024.2.1

declare -a CASES=(
    "eulerian_diffusion_freefree_1d"
    "eulerian_diffusion_freefree_1d_512_limited"
    "eulerian_diffusion_freefree_1d_32"
    "eulerian_diffusion_freefree_1d_32_limited"
)

run_case() {
    local case_id="$1"
    local case_rel="regression_tests/cases/${case_id}"
    local case_dir="${ROOT_DIR}/${case_rel}"
    local build_stdout="${case_dir}/build.stdout.log"
    local build_stderr="${case_dir}/build.stderr.log"
    local run_stdout="${case_dir}/run.stdout.log"
    local run_stderr="${case_dir}/run.stderr.log"
    local rich_bin="${ROOT_DIR}/build/${CONFIG}/${case_id}/rich"
    local case_ntasks="${SLURM_NTASKS_512}"

    if [[ ! -d "${case_dir}" ]]; then
        echo "Missing regression case directory: ${case_dir}" >&2
        exit 2
    fi

    rm -f \
        "${case_dir}/temperature_vs_x"*.png "${case_dir}/temperature_vs_x"*.pdf \
        "${case_dir}/trad_vs_x"*.png "${case_dir}/trad_vs_x"*.pdf \
        "${case_dir}/density_vs_x"*.png "${case_dir}/density_vs_x"*.pdf \
        "${case_dir}/velocity_vs_x"*.png "${case_dir}/velocity_vs_x"*.pdf

    echo "[BUILD] ${case_id}"
    (
        cd "${ROOT_DIR}"
        "${BUILD_SCRIPT}" "${CONFIG}" \
            "--test_name=${case_rel}" \
            "--build-subdir=${case_id}"
    ) >"${build_stdout}" 2>"${build_stderr}"

    if [[ ! -x "${rich_bin}" && ! -L "${rich_bin}" ]]; then
        echo "Expected built binary missing: ${rich_bin}" >&2
        exit 1
    fi

    echo "[RUN] ${case_id}"
    if [[ "${case_id}" == *"_32"* ]]; then
        case_ntasks="${SLURM_NTASKS_32}"
    fi
    if [[ "${RUN_LOCAL}" -eq 1 ]]; then
        (cd "${case_dir}" && mpirun -np "${case_ntasks}" "${rich_bin}") \
            >"${run_stdout}" 2>"${run_stderr}"
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
        "${sbatch_args[@]}"
    fi

    if [[ ! -s "${case_dir}/temperature_profile.txt" ]]; then
        echo "Missing profile output for ${case_id}: ${case_dir}/temperature_profile.txt" >&2
        exit 1
    fi
    if [[ ! -s "${case_dir}/shock_position.txt" ]]; then
        echo "Missing shock output for ${case_id}: ${case_dir}/shock_position.txt" >&2
        exit 1
    fi
}

for case_id in "${CASES[@]}"; do
    run_case "${case_id}"
done

PROFILE_512="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_1d/temperature_profile.txt"
PROFILE_512_LIMITED="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_1d_512_limited/temperature_profile.txt"
PROFILE_32="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_1d_32/temperature_profile.txt"
PROFILE_32_LIMITED="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_1d_32_limited/temperature_profile.txt"
OUTPUT_DIR="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_compare"

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

echo "Comparison plots generated in: ${OUTPUT_DIR}"
