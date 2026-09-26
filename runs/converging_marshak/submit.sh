#!/bin/bash
#SBATCH --job-name=converging-marshak
#SBATCH --partition=bigrun
#SBATCH --ntasks=192
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --output=marshak-%A-%a.out
#SBATCH --error=marshak-%A-%a.err

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

# The bath between the sphere and the box only has to hold the boundary
# temperature, which the thermostat driver shells already do.  Its background
# generators are nevertheless the hottest emitting material in the problem at
# early times, so nearly every packet is born there: raising their count to
# 20000 multiplied the packet budget by an order of magnitude and made a single
# transport sweep take longer than the whole run should.  Keep the bath coarse
# and spend the cells on the radial mesh instead.
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

echo "Converging Marshak test ${TEST} results: ${RESULT_DIR}"
