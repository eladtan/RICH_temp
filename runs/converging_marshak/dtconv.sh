#!/bin/bash
#SBATCH --job-name=marshak-dtconv
#SBATCH --partition=bigrun
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --output=dtconv-%j.out
#SBATCH --error=dtconv-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
RESULT_DIR="${SLURM_SUBMIT_DIR}/results/dtconv"
mkdir -p "${RESULT_DIR}"
mpirun -np "${SLURM_NTASKS}" ./rich 3 --cells-per-mfp 1 --min-shells 16 --angular-points 200 \
    --exterior-points 2000 --front-cells 0.075 --new-photons 2 --population 20 --seed 1850 \
    --output-dir "${RESULT_DIR}"
python3 ./compare_profiles.py --test 3 --input-dir "${RESULT_DIR}" --output-dir "${RESULT_DIR}" --bins 48
echo "DTCONV DONE"
