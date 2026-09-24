#!/bin/bash
# Same 256×32000 mesh as marshak-t2-4xyz (job 10145415), with DDMC on.
#SBATCH --time=48:00:00
#SBATCH --ntasks=384
#SBATCH --output=marshak-tN-4xyz-ddmc-%j.out
#SBATCH --error=marshak-tN-4xyz-ddmc-%j.err

set -euo pipefail

if [[ -z "${TEST:-}" ]]; then
    echo "Set TEST (1-4) via sbatch --export=ALL,TEST=N" >&2
    exit 1
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
RICH_BIN="${RICH_BIN:-./rich_ddmc}"
if [[ ! -x "${SCRIPT_DIR}/${RICH_BIN#./}" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/${RICH_BIN#./}" >&2
    exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/shared/maorm/CovnergingMarshak/Benchmark${TEST}}"
mkdir -p "${CHECKPOINT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    RESUME_ARGS=(--resume "${RESUME}")
    echo "Resuming test ${TEST} from ${RESUME} on ${SLURM_NTASKS} ranks"
fi

VTK_CYCLES="${VTK_CYCLES:-200}"
HDF5_CYCLES="${HDF5_CYCLES:-0}"
LATEST_CYCLES="${LATEST_CYCLES:-200}"

mpirun -np "${SLURM_NTASKS}" "${RICH_BIN}" "${TEST}" \
    --cells-per-mfp 4 \
    --min-shells 256 \
    --max-shells 800 \
    --angular-points 32000 \
    --ddmc true \
    --output "${CHECKPOINT_DIR}" \
    --hdf5-cycles "${HDF5_CYCLES}" \
    --vtk-cycles "${VTK_CYCLES}" \
    --latest-cycles "${LATEST_CYCLES}" \
    --output-dir "${CHECKPOINT_DIR}" \
    "${RESUME_ARGS[@]}"

if [[ -f "${CHECKPOINT_DIR}/test${TEST}_snapshot3.csv" ]]; then
    python3 ./compare_profiles.py --test "${TEST}" --input-dir "${CHECKPOINT_DIR}" --output-dir "${CHECKPOINT_DIR}" --bins 96 --plot-mode binned
fi

echo "Converging Marshak test ${TEST} 4x-xyz DDMC (${SLURM_NTASKS} ranks) results: ${CHECKPOINT_DIR}"
