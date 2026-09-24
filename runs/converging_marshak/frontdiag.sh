#!/bin/bash
#SBATCH --job-name=marshak-frontdiag
#SBATCH --partition=bigrun
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=frontdiag-%A-%a.out
#SBATCH --error=frontdiag-%A-%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1

# Tasks 1-4 resolve the front with the mean-free-path based radial mesh.
# Task 5 repeats test 1 on the old eight-cell uniform mesh, so that the
# difference between the two isolates the effect of front resolution alone.
case "${SLURM_ARRAY_TASK_ID}" in
    1) TEST=1; NAME=mfp1;    ARGS="--cells-per-mfp 1 --min-shells 16 --angular-points 200" ;;
    2) TEST=2; NAME=mfp2;    ARGS="--cells-per-mfp 1 --min-shells 16 --angular-points 200" ;;
    3) TEST=3; NAME=mfp3;    ARGS="--cells-per-mfp 1 --min-shells 16 --angular-points 200" ;;
    4) TEST=4; NAME=mfp4;    ARGS="--cells-per-mfp 1 --min-shells 16 --angular-points 200" ;;
    5) TEST=1; NAME=coarse1; ARGS="--min-shells 8 --max-shells 8 --angular-points 200" ;;
esac

RESULT_DIR="${SLURM_SUBMIT_DIR}/results/frontdiag/${NAME}"
mkdir -p "${RESULT_DIR}"

mpirun -np "${SLURM_NTASKS}" ./rich "${TEST}" \
    ${ARGS} \
    --exterior-points 20000 \
    --new-photons 2 \
    --population 20 \
    --seed "$((1847 + SLURM_ARRAY_TASK_ID))" \
    --output-dir "${RESULT_DIR}"

python3 ./compare_profiles.py \
    --test "${TEST}" \
    --input-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}" \
    --bins 32

echo "FRONTDIAG ${NAME} DONE"
