#!/bin/bash
#SBATCH --job-name=marshak-accurate
#SBATCH --partition=bigrun
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --output=accurate-%A-%a.out
#SBATCH --error=accurate-%A-%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ ! -x "${SCRIPT_DIR}/rich" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/rich" >&2
    exit 1
fi

TEST="${SLURM_ARRAY_TASK_ID}"
RESULT_DIR="${SCRIPT_DIR}/results/accurate/test${TEST}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

# Accuracy recipe: front-limited timestep (--front-cells 0.15), two cells per
# heated mean free path, high angular count, large packet population.
mpirun -np "${SLURM_NTASKS}" ./rich "${TEST}" \
    --cells-per-mfp 2 \
    --min-shells 16 \
    --angular-points 400 \
    --exterior-points 2000 \
    --front-cells 0.15 \
    --new-photons 8 \
    --population 80 \
    --seed "$((2847 + TEST))" \
    --output-dir "${RESULT_DIR}"

python3 ./compare_profiles.py \
    --test "${TEST}" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --bins 96

echo "Accurate Marshak test ${TEST} results: ${RESULT_DIR}"
