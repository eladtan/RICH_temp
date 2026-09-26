#!/bin/bash
#SBATCH --job-name=rich-krti
#SBATCH --partition=bigrun
#SBATCH --ntasks=80
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --time=21-00:00:00
#SBATCH --output=krti-%j.out
#SBATCH --error=krti-%j.err

set -euo pipefail

SCRIPT_DIR=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/source/monte/CMakeLists.txt" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/runs/Mizrachi_KRTI"
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
EXECUTABLE="${KRTI_EXECUTABLE:-${SCRIPT_DIR}/rich}"

KRTI_CELLS_VALUE="${KRTI_CELLS_PER_LAMBDA:-32}"
KRTI_K_ETA0_VALUE="${KRTI_K_ETA0:-0.001}"
KRTI_INITIAL_VALUE="${KRTI_INITIAL_PARTICLES:-4000}"
KRTI_NEW_VALUE="${KRTI_NEW_PHOTONS:-4000}"
KRTI_POPULATION_VALUE="${KRTI_POPULATION:-4000}"
KRTI_FINAL_RT_VALUE="${KRTI_FINAL_RT:-2.0}"
KRTI_OUTPUT_CYCLES_VALUE="${KRTI_OUTPUT_CYCLES:-500}"
KRTI_ARCHIVE_CYCLES_VALUE="${KRTI_ARCHIVE_CYCLES:-1000000000}"
KRTI_VTK_CYCLES_VALUE="${KRTI_VTK_CYCLES:-1000000000}"
KRTI_REFERENCE_DIR="${KRTI_REFERENCE_DIR:-${SCRIPT_DIR}/KRTI_S_X_reference_package/reference}"
KRTI_SEED_VALUE="${KRTI_SEED:-20260828}"
RESULT_DIR="${KRTI_OUTPUT_DIR:-${SCRIPT_DIR}/results/${SLURM_JOB_ID}}"
HISTORY="${RESULT_DIR}/krti_history.csv"

RATE_WINDOW_ARGS=()
if [[ -n "${KRTI_RATE_MIN:-}" || -n "${KRTI_RATE_MAX:-}" ]]; then
    if [[ -z "${KRTI_RATE_MIN:-}" || -z "${KRTI_RATE_MAX:-}" ]]; then
        echo "KRTI_RATE_MIN and KRTI_RATE_MAX must be supplied together" >&2
        exit 2
    fi
    RATE_WINDOW_ARGS=(--rate-window "${KRTI_RATE_MIN}" "${KRTI_RATE_MAX}")
fi
RATE_TOLERANCE_ARGS=()
if [[ -n "${KRTI_RATE_RELATIVE_TOLERANCE:-}" ]]; then
    RATE_TOLERANCE_ARGS=(--rate-relative-tolerance "${KRTI_RATE_RELATIVE_TOLERANCE}")
fi
RESTART_ARGS=()
if [[ -n "${KRTI_RESTART:-}" ]]; then
    RESTART_ARGS=(--restart "${KRTI_RESTART}")
fi

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Missing KRTI executable at ${EXECUTABLE}." >&2
    echo "Build it before submitting:" >&2
    echo "  cd ${RICH_ROOT} && ./build_rich.sh gnuReleaseMPI --test_name=Mizrachi_KRTI --build-subdir=krti --jobs=16" >&2
    echo "  cp ${RICH_ROOT}/build/gnuReleaseMPI/krti/rich_gnuReleaseMPI ${SCRIPT_DIR}/rich" >&2
    exit 2
fi

mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

MPI_TASKS="${SLURM_NTASKS:-80}"
mpirun -np "${MPI_TASKS}" "${EXECUTABLE}" \
    --case X \
    --manager "${KRTI_MANAGER:-rdma}" \
    --cells-per-lambda "${KRTI_CELLS_VALUE}" \
    --k-eta0 "${KRTI_K_ETA0_VALUE}" \
    --initial-particles "${KRTI_INITIAL_VALUE}" \
    --new-photons "${KRTI_NEW_VALUE}" \
    --population "${KRTI_POPULATION_VALUE}" \
    --final-time "${KRTI_FINAL_RT_VALUE}" \
    --output-cycles "${KRTI_OUTPUT_CYCLES_VALUE}" \
    --archive-cycles "${KRTI_ARCHIVE_CYCLES_VALUE}" \
    --vtk-cycles "${KRTI_VTK_CYCLES_VALUE}" \
    --reference-pressure 1.0e12 \
    --reference-directory "${KRTI_REFERENCE_DIR}" \
    --seed "${KRTI_SEED_VALUE}" \
    --output "${RESULT_DIR}" \
    "${RESTART_ARGS[@]}"

python3 "${SCRIPT_DIR}/compare_krti.py" \
    --input "${HISTORY}" \
    --plot "${RESULT_DIR}/krti_comparison.png" \
    --json "${RESULT_DIR}/krti_metrics.json" \
    "${RATE_WINDOW_ARGS[@]}" \
    "${RATE_TOLERANCE_ARGS[@]}"

echo "KRTI results: ${RESULT_DIR}"
