#!/bin/bash
#SBATCH --job-name=rich-tre1d
#SBATCH --partition=bigrun
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=tre-1d-%j.out
#SBATCH --error=tre-1d-%j.err

set -euo pipefail

SCRIPT_DIR=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/source/monte/CMakeLists.txt" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/runs/Mizrachi_TRE_benchmark"
    elif [[ -f "${SLURM_SUBMIT_DIR}/CMakeLists.txt" &&
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
BUILD_DIR="${RICH_ROOT}/build/gnuReleaseMPI"
RESULT_DIR="${SCRIPT_DIR}/results/${SLURM_JOB_ID:-local}"
HISTORY="${RESULT_DIR}/tre_1d_history.csv"
MPI_RANKS="${SLURM_NTASKS:-16}"

if [[ ! -x "${SCRIPT_DIR}/rich" ||
      "${SCRIPT_DIR}/test.cpp" -nt "${SCRIPT_DIR}/rich" ||
      "${RICH_ROOT}/source/3D/tessellation/cartesian/CartesianMesh3D.hpp" -nt "${SCRIPT_DIR}/rich" ||
      "${RICH_ROOT}/source/monte/utils/GhostMap.hpp" -nt "${SCRIPT_DIR}/rich" ]]; then
    echo "Building RICH executable for Mizrachi_TRE_benchmark..." >&2
    (
        cd "${RICH_ROOT}"
        ./build_rich.sh gnuReleaseMPI --test_name=Mizrachi_TRE_benchmark --jobs="${SLURM_CPUS_PER_TASK:-1}"
    )
    cp -f "${BUILD_DIR}/rich_gnuReleaseMPI" "${SCRIPT_DIR}/rich"
fi

mkdir -p "${RESULT_DIR}"
export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

mpirun ./rich \
    --nx 1024 \
    --initial-particles 1000 \
    --new-photons 50 \
    --population 1000 \
    --dtau 0.005 \
    --final-tau 10.0 \
    --output "${HISTORY}"

python3 "${SCRIPT_DIR}/analyze_tre_1d.py" \
    --input "${HISTORY}" \
    --plot "${RESULT_DIR}/tre_1d_comparison.png" \
    --json "${RESULT_DIR}/tre_1d_metrics.json"

echo "TRE-1 results: ${RESULT_DIR}"
