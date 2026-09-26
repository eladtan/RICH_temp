#!/bin/bash
#SBATCH --job-name=marshak-t4
#SBATCH --partition=bigrun
#SBATCH --ntasks=192
#SBATCH --time=24:00:00
#SBATCH --exclude=d25g[73-84]
#SBATCH --output=marshak-t%j.out
#SBATCH --error=marshak-t%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ ! -x "${SCRIPT_DIR}/rich" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/rich" >&2
    exit 1
fi

TEST=4
RESULT_DIR="${SCRIPT_DIR}/results/${SLURM_JOB_ID}/test${TEST}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

# DDMC off: stalls converging fronts in cold interior. Halved packet stats for speed.
mpirun -np "${SLURM_NTASKS}" ./rich "${TEST}" \
    --cells-per-mfp 2 \
    --min-shells 16 \
    --angular-points 200 \
    --exterior-points 2000 \
    --front-cells 0.15 \
    --new-photons 1 \
    --population 10 \
    --ddmc false \
    --seed "$((1847 + TEST))" \
    --output-dir "${RESULT_DIR}"

python3 ./compare_profiles.py \
    --test "${TEST}" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --bins 96

echo "Converging Marshak test ${TEST} (standard radial, no DDMC, ${SLURM_NTASKS} ranks) results: ${RESULT_DIR}"
