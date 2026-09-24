#!/bin/bash
#SBATCH --job-name=jds-g3-imc
#SBATCH --partition=bigrun
#SBATCH --ntasks=160
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --time=21-00:00:00
#SBATCH --output=jds-g3-%j.out
#SBATCH --error=jds-g3-%j.err

set -euo pipefail

SCRIPT_DIR=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/source/monte/CMakeLists.txt" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/runs/Jiang_Davis_Stone"
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
EXECUTABLE="${JDS_EXECUTABLE:-${SCRIPT_DIR}/rich}"

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Missing G3 IMC executable at ${EXECUTABLE}." >&2
    echo "Build it before submitting:" >&2
    echo "  cd ${RICH_ROOT} && ./build_rich.sh gnuReleaseMPI --test_name=Jiang_Davis_Stone --build-subdir=jds_g3 --jobs=16" >&2
    echo "  cp ${RICH_ROOT}/build/gnuReleaseMPI/jds_g3/rich_gnuReleaseMPI ${SCRIPT_DIR}/rich" >&2
    exit 2
fi

RESULT_DIR="${JDS_OUTPUT_DIR:-}"
if [[ -z "${RESULT_DIR}" ]]; then
    if [[ -n "${JDS_RESTART:-}" ]]; then
        RESULT_DIR="$(cd "$(dirname "${JDS_RESTART}")" && pwd)"
        if [[ "$(basename "${RESULT_DIR}")" == "snapshots" ]]; then
            RESULT_DIR="$(cd "${RESULT_DIR}/.." && pwd)"
        fi
    else
        RESULT_DIR="/data/shared/maorm/JDS/${SLURM_JOB_ID}"
    fi
fi
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

RESTART_ARGS=()
if [[ -n "${JDS_RESTART:-}" ]]; then
    RESTART_ARGS=(--restart "${JDS_RESTART}")
fi

HISTORY="${RESULT_DIR}/jds_g3_history.csv"

MPI_TASKS="${SLURM_NTASKS:-160}"
mpirun -np "${MPI_TASKS}" "${EXECUTABLE}" \
    --manager "${JDS_MANAGER:-rdma}" \
    --nx "${JDS_NX:-128}" \
    --ny 1 \
    --nz "${JDS_NZ:-512}" \
    --gamma "${JDS_GAMMA:-1.6666666666666667}" \
    --seed "${JDS_SEED:-20260831}" \
    --initial-particles "${JDS_INITIAL_PARTICLES:-4000}" \
    --boundary-particles "${JDS_BOUNDARY_PARTICLES:-4000}" \
    --population "${JDS_POPULATION:-4000}" \
    --final-time "${JDS_FINAL_TIME:-28.9}" \
    --history-every "${JDS_HISTORY_EVERY:-5}" \
    --output-cycles "${JDS_OUTPUT_CYCLES:-500}" \
    --checkpoint-cycles "${JDS_CHECKPOINT_CYCLES:-5000}" \
    --vtk-cycles "${JDS_VTK_CYCLES:-1000}" \
    --output "${RESULT_DIR}" \
    "${RESTART_ARGS[@]}"

python3 "${SCRIPT_DIR}/compare_jds.py" \
    --input "${HISTORY}" \
    --json "${RESULT_DIR}/jds_g3_metrics.json" \
    --plot "${RESULT_DIR}/jds_g3_comparison.png"

echo "JDS G3 IMC results: ${RESULT_DIR}"
