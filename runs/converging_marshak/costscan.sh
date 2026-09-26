#!/bin/bash
#SBATCH --job-name=marshak-costscan
#SBATCH --partition=bigrun
#SBATCH --ntasks=16
#SBATCH --time=01:30:00
#SBATCH --array=1-4
#SBATCH --output=costscan-%A-%a.out
#SBATCH --error=costscan-%A-%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1

# Test 1 has a Fleck factor of about 2e-3, so a packet pays several hundred
# effective scatters per true absorption.  On a front-resolving mesh the cells
# are about one mean free path thick, which is below the random walk threshold,
# so none of that is accelerated.  These four variants measure the cost of the
# available remedies on a deliberately small mesh.
case "${SLURM_ARRAY_TASK_ID}" in
    1) NAME=control;  ARGS="--cells-per-mfp 1" ;;
    2) NAME=ddmc;     ARGS="--cells-per-mfp 1 --ddmc true" ;;
    3) NAME=lowrw;    ARGS="--cells-per-mfp 1 --rw-cell-depth 0.5 --rw-particle-depth 0.2" ;;
    4) NAME=thickmfp; ARGS="--cells-per-mfp 0.33" ;;
esac

RESULT_DIR="${SLURM_SUBMIT_DIR}/results/costscan/${NAME}"
mkdir -p "${RESULT_DIR}"

STEPS=10
START=${SECONDS}
mpirun -np "${SLURM_NTASKS}" ./rich 1 \
    ${ARGS} \
    --min-shells 16 \
    --angular-points 24 \
    --exterior-points 2000 \
    --new-photons 2 \
    --population 20 \
    --steps "${STEPS}" \
    --output-dir "${RESULT_DIR}" || echo "VARIANT ${NAME} FAILED"
ELAPSED=$((SECONDS - START))

echo "COSTSCAN ${NAME}: ${ELAPSED} s for ${STEPS} steps -> $((ELAPSED / STEPS)) s/step"
