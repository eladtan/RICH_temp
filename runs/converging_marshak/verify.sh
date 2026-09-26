#!/bin/bash
#SBATCH --job-name=marshak-verify
#SBATCH --partition=bigrun
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --array=1-4
#SBATCH --output=verify-%A-%a.out
#SBATCH --error=verify-%A-%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
TEST="${SLURM_ARRAY_TASK_ID}"
RESULT_DIR="${SCRIPT_DIR}/results/verify/test${TEST}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

mpirun -np "${SLURM_NTASKS}" ./rich "${TEST}" \
    --radial-shells 16 \
    --exterior-points 4000 \
    --new-photons 1 \
    --population 8 \
    --steps 40 \
    --seed "$((1847 + TEST))" \
    --output-dir "${RESULT_DIR}"

python3 ./compare_profiles.py \
    --test "${TEST}" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --bins 32

echo "Converging Marshak verification ${TEST}: ${RESULT_DIR}"
