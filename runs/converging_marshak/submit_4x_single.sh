#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --time=48:00:00
#SBATCH --output=marshak-4x-t%j.out
#SBATCH --error=marshak-4x-t%j.err

set -euo pipefail

if [[ -z "${TEST:-}" ]]; then
    echo "Set TEST (1-4) via sbatch --export=ALL,TEST=N" >&2
    exit 1
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
# RICH_BIN lets a new build be tested while older jobs still run from ./rich;
# overwriting a mapped executable is refused by the kernel.
RICH_BIN="${RICH_BIN:-./rich}"
if [[ ! -x "${SCRIPT_DIR}/${RICH_BIN#./}" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/${RICH_BIN#./}" >&2
    exit 1
fi
BATH_GRADING="${BATH_GRADING:-0}"
DDMC="${DDMC:-false}"

CHECKPOINT_DIR="/data/shared/maorm/CovnergingMarshak/Benchmark${TEST}"
mkdir -p "${CHECKPOINT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

# Tests 1-3: 3x more radial shells than the previous 4x runs (cells-per-mfp 8).
# Test 4 keeps the 4x radial mesh.
# Four times as many generators per shell (200 -> 800) with packets per cell
# cut by the same factor, so the Monte Carlo work per step stays about the same
# while neighbouring directions share a smoother spherical mesh.
if [[ "${TEST}" -eq 4 ]]; then
    CELLS_PER_MFP=8
    MAX_SHELLS=1600
else
    CELLS_PER_MFP=24
    MAX_SHELLS=8000
fi
ANGULAR_POINTS="${ANGULAR_POINTS:-800}"
NEW_PHOTONS="${NEW_PHOTONS:-1}"
POPULATION="${POPULATION:-3}"

mpirun -np "${SLURM_NTASKS}" "${RICH_BIN}" "${TEST}" \
    --cells-per-mfp "${CELLS_PER_MFP}" \
    --min-shells 16 \
    --max-shells "${MAX_SHELLS}" \
    --angular-points "${ANGULAR_POINTS}" \
    --exterior-points 2000 \
    --bath-grading "${BATH_GRADING}" \
    --front-cells 0.15 \
    --new-photons "${NEW_PHOTONS}" \
    --population "${POPULATION}" \
    --ddmc "${DDMC}" \
    --seed "$((4847 + TEST))" \
    --output "${CHECKPOINT_DIR}" \
    --hdf5-cycles 500 \
    --vtk-cycles 200 \
    --output-dir "${CHECKPOINT_DIR}"

python3 ./compare_profiles.py \
    --test "${TEST}" \
    --input-dir "${CHECKPOINT_DIR}" \
    --output-dir "${CHECKPOINT_DIR}" \
    --bins 96

echo "Converging Marshak test ${TEST} (${SLURM_NTASKS} ranks) results: ${CHECKPOINT_DIR}"
