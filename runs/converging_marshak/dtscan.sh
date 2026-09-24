#!/bin/bash
#SBATCH --job-name=marshak-dtscan
#SBATCH --partition=bigrun
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --array=1-3
#SBATCH --output=dtscan-%A-%a.out
#SBATCH --error=dtscan-%A-%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1

# Test 3 ran with time steps 6.5x larger than RadiationMCStep::suggestTimeStep()
# asked for, at every step, and its wave lags the similarity solution by half a
# micron.  --front-cells scales the step directly, so 1/6.5 reproduces the
# suggested step without touching the code.  Variant 3 raises the packet count at
# the same time to separate time resolution from statistics.
case "${SLURM_ARRAY_TASK_ID}" in
    1) NAME=fc050; ARGS="--front-cells 0.5  --new-photons 2 --population 20" ;;
    2) NAME=fc015; ARGS="--front-cells 0.15 --new-photons 2 --population 20" ;;
    3) NAME=fc015x4; ARGS="--front-cells 0.15 --new-photons 8 --population 80" ;;
esac

RESULT_DIR="${SLURM_SUBMIT_DIR}/results/dtscan/${NAME}"
mkdir -p "${RESULT_DIR}"

mpirun -np "${SLURM_NTASKS}" ./rich 3 \
    --cells-per-mfp 1 \
    --min-shells 16 \
    --angular-points 200 \
    --exterior-points 2000 \
    ${ARGS} \
    --output-dir "${RESULT_DIR}"

python3 ./compare_profiles.py \
    --test 3 \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --bins 48

echo "DTSCAN ${NAME} DONE"
