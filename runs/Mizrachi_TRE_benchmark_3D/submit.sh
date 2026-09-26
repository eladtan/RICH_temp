#!/bin/bash
#SBATCH --job-name=rich-tre3d
#SBATCH --partition=bigrun
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=tre-3d-%j.out
#SBATCH --error=tre-3d-%j.err

set -euo pipefail

SCRIPT_DIR=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # Slurm executes a copied batch script from its spool directory.  Resolve
    # this benchmark from the submit directory instead of BASH_SOURCE[0].
    if [[ -f "${SLURM_SUBMIT_DIR}/source/monte/CMakeLists.txt" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/runs/Mizrachi_TRE_benchmark_3D"
    elif [[ -f "${SLURM_SUBMIT_DIR}/main.cpp" &&
            -f "${SLURM_SUBMIT_DIR}/../../source/monte/CMakeLists.txt" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    fi
fi
if [[ -z "${SCRIPT_DIR}" ]]; then
    SCRIPT_SOURCE="${BASH_SOURCE[0]}"
    if [[ "${SCRIPT_SOURCE}" != /* ]]; then
        SCRIPT_SOURCE="${SLURM_SUBMIT_DIR:-${PWD}}/${SCRIPT_SOURCE}"
    fi
    SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
fi
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
RICH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXECUTABLE="${TRE_EXECUTABLE:-${SCRIPT_DIR}/rich}"

TRE_N_VALUE="${TRE_N:-16}"
TRE_INITIAL_VALUE="${TRE_INITIAL_PARTICLES:-300}"
TRE_NEW_VALUE="${TRE_NEW_PHOTONS:-8}"
TRE_POPULATION_VALUE="${TRE_POPULATION:-800}"
TRE_DTAU_VALUE="${TRE_DTAU:-0.01}"
TRE_FINAL_VALUE="${TRE_FINAL_TAU:-8.0}"
TRE_SEED_VALUE="${TRE_SEED:-20260828}"
TRE_SNAPSHOT_EVERY_VALUE="${TRE_SNAPSHOT_EVERY:-100}"
TRE_OUTPUT_ROOT_VALUE="${TRE_OUTPUT_ROOT:-/data/shared/maorm/TRE3D}"
RESULT_DIR="${TRE_OUTPUT_DIR:-${TRE_OUTPUT_ROOT_VALUE}/${SLURM_JOB_ID:-local}}"
HISTORY="${RESULT_DIR}/tre_3d_history.csv"

RATE_WINDOW_ARGS=()
if [[ -n "${TRE_RATE_MIN:-}" || -n "${TRE_RATE_MAX:-}" ]]; then
    if [[ -z "${TRE_RATE_MIN:-}" || -z "${TRE_RATE_MAX:-}" ]]; then
        echo "TRE_RATE_MIN and TRE_RATE_MAX must be supplied together" >&2
        exit 2
    fi
    RATE_WINDOW_ARGS=(--rate-window "${TRE_RATE_MIN}" "${TRE_RATE_MAX}")
fi
RATE_TOLERANCE_ARGS=()
if [[ -n "${TRE_RATE_RELATIVE_TOLERANCE:-}" ]]; then
    RATE_TOLERANCE_ARGS=(--rate-relative-tolerance "${TRE_RATE_RELATIVE_TOLERANCE}")
fi

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Missing TRE-3D executable at ${EXECUTABLE}." >&2
    echo "Build it before submitting:" >&2
    echo "  cd ${RICH_ROOT} && ./build_rich.sh gnuReleaseMPI --test_name=Mizrachi_TRE_benchmark_3D --build-subdir=tre_3d --jobs=16" >&2
    echo "  cp ${RICH_ROOT}/build/gnuReleaseMPI/tre_3d/rich_gnuReleaseMPI ${SCRIPT_DIR}/rich" >&2
    exit 2
fi

mkdir -p "${RESULT_DIR}"

mpirun "${EXECUTABLE}" \
    --n "${TRE_N_VALUE}" \
    --initial-particles "${TRE_INITIAL_VALUE}" \
    --new-photons "${TRE_NEW_VALUE}" \
    --population "${TRE_POPULATION_VALUE}" \
    --dtau "${TRE_DTAU_VALUE}" \
    --final-tau "${TRE_FINAL_VALUE}" \
    --snapshot-every "${TRE_SNAPSHOT_EVERY_VALUE}" \
    --seed "${TRE_SEED_VALUE}" \
    --output "${RESULT_DIR}"

python3 "${SCRIPT_DIR}/analyze_tre_3d.py" \
    --input "${HISTORY}" \
    --plot "${RESULT_DIR}/tre_3d_comparison.png" \
    --json "${RESULT_DIR}/tre_3d_metrics.json" \
    "${RATE_WINDOW_ARGS[@]}" \
    "${RATE_TOLERANCE_ARGS[@]}"

echo "TRE-3D results: ${RESULT_DIR}"
