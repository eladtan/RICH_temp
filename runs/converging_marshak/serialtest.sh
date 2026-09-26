#!/bin/bash
#SBATCH --job-name=marshak-serial
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=serialtest-%j.out
#SBATCH --error=serialtest-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
mpirun -np 1 ./rich 1 --radial-shells 16 --exterior-points 4000 \
    --new-photons 1 --population 8 --steps 40 \
    --output-dir "${SLURM_SUBMIT_DIR}/results/serial1"
echo "SERIALTEST DONE"
