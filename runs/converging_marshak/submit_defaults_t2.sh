#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --time=48:00:00
#SBATCH --ntasks=768
#SBATCH --job-name=marshak-t2-4xyz
#SBATCH --output=marshak-t2-4xyz-%j.out
#SBATCH --error=marshak-t2-4xyz-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
RICH_BIN="${RICH_BIN:-./rich_defaults}"
if [[ ! -x "${SCRIPT_DIR}/${RICH_BIN#./}" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/${RICH_BIN#./}" >&2
    exit 1
fi

CHECKPOINT_DIR="/data/shared/maorm/CovnergingMarshak/Benchmark2"
mkdir -p "${CHECKPOINT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

# 2x linear resolution in r, theta, and phi vs job 10145282 (128 shells x 8000
# angular): 256 shells and 32000 generators per shell. Same IMC/no-DDMC physics.
mpirun -np "${SLURM_NTASKS}" "${RICH_BIN}" 2 \
    --cells-per-mfp 4 \
    --min-shells 256 \
    --max-shells 800 \
    --angular-points 32000 \
    --ddmc false \
    --output "${CHECKPOINT_DIR}" \
    --hdf5-cycles 500 \
    --vtk-cycles 200 \
    --output-dir "${CHECKPOINT_DIR}"

if [[ -f "${CHECKPOINT_DIR}/test2_snapshot3.csv" ]]; then
    python3 ./compare_profiles.py --test 2 --input-dir "${CHECKPOINT_DIR}" --output-dir "${CHECKPOINT_DIR}" --bins 96 --plot-mode binned
fi

echo "Converging Marshak test 2 4x-xyz IMC (${SLURM_NTASKS} ranks) results: ${CHECKPOINT_DIR}"
