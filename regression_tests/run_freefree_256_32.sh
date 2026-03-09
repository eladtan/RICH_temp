#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGRESSION_DIR="${ROOT_DIR}/regression_tests"
RUNNER="${REGRESSION_DIR}/run_all.sh"
COMPARE_PLOTTER="${REGRESSION_DIR}/lib/plot_eulerian_diffusion_freefree_compare.py"

CONFIG="${1:-intelReleaseMPI}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${RUNNER}" ]]; then
    echo "Missing executable runner: ${RUNNER}" >&2
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

ml openmpi/4.1.6/Intel/OneApi/2024.2.1

"${RUNNER}" --test eulerian_diffusion_freefree_1d --config "${CONFIG}" --keep-artifacts
"${RUNNER}" --test eulerian_diffusion_freefree_1d_32 --config "${CONFIG}" --keep-artifacts

PROFILE_256="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_1d/temperature_profile.txt"
PROFILE_32="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_1d_32/temperature_profile.txt"
OUTPUT_DIR="${REGRESSION_DIR}/cases/eulerian_diffusion_freefree_compare"

if [[ ! -s "${PROFILE_256}" ]]; then
    echo "Missing profile output for 256-cell case: ${PROFILE_256}" >&2
    exit 1
fi

if [[ ! -s "${PROFILE_32}" ]]; then
    echo "Missing profile output for 32-cell case: ${PROFILE_32}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${COMPARE_PLOTTER}" \
    --profile-256 "${PROFILE_256}" \
    --profile-32 "${PROFILE_32}" \
    --output-dir "${OUTPUT_DIR}"

echo "Comparison plots generated in: ${OUTPUT_DIR}"
