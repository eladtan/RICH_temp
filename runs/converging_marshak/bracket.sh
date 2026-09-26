#!/bin/bash
#SBATCH --job-name=marshak-bracket
#SBATCH --partition=bigrun
#SBATCH --ntasks=32
#SBATCH --time=00:25:00
#SBATCH --array=1-4
#SBATCH --output=bracket-%A-%a.out
#SBATCH --error=bracket-%A-%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1

# The 24-angular-point scan runs 10 steps in 30 s, while the 200-angular-point
# production mesh did not finish a single transport sweep in 14 minutes.  These
# variants change one knob at a time to find which one drives the cost.  The
# "Loop time / max steps" line reports transport events per sweep directly.
case "${SLURM_ARRAY_TASK_ID}" in
    1) NAME=a200_e20000; ANG=200; EXT=20000 ;;
    2) NAME=a200_e2000;  ANG=200; EXT=2000  ;;
    3) NAME=a60_e20000;  ANG=60;  EXT=20000 ;;
    4) NAME=a60_e2000;   ANG=60;  EXT=2000  ;;
esac

RESULT_DIR="${SLURM_SUBMIT_DIR}/results/bracket/${NAME}"
mkdir -p "${RESULT_DIR}"

mpirun -np "${SLURM_NTASKS}" ./rich 1 \
    --cells-per-mfp 1 \
    --min-shells 16 \
    --angular-points "${ANG}" \
    --exterior-points "${EXT}" \
    --new-photons 2 \
    --population 20 \
    --steps 10 \
    --output-dir "${RESULT_DIR}" || true

echo "BRACKET ${NAME} DONE"
