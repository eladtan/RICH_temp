#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --time=12:00:00
#SBATCH --output=marshak-t%j.out
#SBATCH --error=marshak-t%j.err

set -euo pipefail

if [[ -z "${TEST:-}" ]]; then
    echo "Set TEST (1-4) via sbatch --export=ALL,TEST=N" >&2
    exit 1
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ ! -x "${SCRIPT_DIR}/rich" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/rich" >&2
    exit 1
fi

RESULT_DIR="${SCRIPT_DIR}/results/${SLURM_JOB_ID}/test${TEST}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

mpirun -np "${SLURM_NTASKS}" ./rich "${TEST}" \
    --cells-per-mfp 2 \
    --min-shells 16 \
    --angular-points 200 \
    --exterior-points 2000 \
    --front-cells 0.15 \
    --new-photons 2 \
    --population 20 \
    --seed "$((1847 + TEST))" \
    --output-dir "${RESULT_DIR}"

python3 ./compare_profiles.py \
    --test "${TEST}" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --bins 96

echo "Converging Marshak test ${TEST} (standard radial, ${SLURM_NTASKS} ranks) results: ${RESULT_DIR}"
