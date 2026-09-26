#!/bin/bash
#SBATCH --job-name=marshak-mgrcheck
#SBATCH --partition=bigrun
#SBATCH --ntasks=16
#SBATCH --time=00:20:00
#SBATCH --output=mgrcheck-%j.out
#SBATCH --error=mgrcheck-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1

mpirun -np "${SLURM_NTASKS}" ./rich 1 \
    --cells-per-mfp 1 \
    --min-shells 16 \
    --angular-points 200 \
    --exterior-points 2000 \
    --new-photons 2 \
    --population 20 \
    --steps 10 \
    --output-dir "${SLURM_SUBMIT_DIR}/results/mgrcheck" || true

echo "MGRCHECK DONE"
