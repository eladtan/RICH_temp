#!/bin/bash
#SBATCH --job-name=marshak-4x-radial
#SBATCH --partition=bigrun
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --output=marshak-4x-%A-%a.out
#SBATCH --error=marshak-4x-%A-%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ ! -x "${SCRIPT_DIR}/rich" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/rich" >&2
    exit 1
fi

TEST="${SLURM_ARRAY_TASK_ID}"
RESULT_DIR="${SCRIPT_DIR}/results/${SLURM_ARRAY_JOB_ID}/test${TEST}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

# Four times as many radial shells as submit.sh (--cells-per-mfp 2): shell width is
# mfp/cells-per-mfp, so quadruple cells-per-mfp to quarter the width (4x shells).
# Raise max-shells so tests 1 and 4 are not capped below 4x (they approach 400 shells at cpmfp 2).
mpirun -np "${SLURM_NTASKS}" ./rich "${TEST}" \
    --cells-per-mfp 8 \
    --min-shells 16 \
    --max-shells 1600 \
    --angular-points 200 \
    --exterior-points 2000 \
    --front-cells 0.15 \
    --new-photons 2 \
    --population 20 \
    --seed "$((3847 + TEST))" \
    --output-dir "${RESULT_DIR}"

python3 ./compare_profiles.py \
    --test "${TEST}" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --bins 96

echo "Converging Marshak test ${TEST} (4x radial shells) results: ${RESULT_DIR}"
