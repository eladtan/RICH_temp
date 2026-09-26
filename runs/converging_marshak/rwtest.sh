#!/bin/bash
#SBATCH --job-name=marshak-rw
#SBATCH --partition=bigrun
#SBATCH --ntasks=16
#SBATCH --time=00:40:00
#SBATCH --output=rwtest-%j.out
#SBATCH --error=rwtest-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
mpirun -np "${SLURM_NTASKS}" ./rich 1 --radial-shells 16 --exterior-points 4000 \
    --new-photons 1 --population 8 --steps 40 \
    --output-dir "${SLURM_SUBMIT_DIR}/results/rw1"
echo "RWTEST DONE"
